import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from .training_monitor import TrainingMonitor

class TrainerGan:
    def __init__(self, loss_fn, accuracy_fn, train_loader: DataLoader, val_loader: DataLoader, model_path: str, device='cpu'):
        self.model_path = model_path
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.monitor = TrainingMonitor()

    def train(self, epochs, generator, discriminator, gen_optimizer, disc_optimizer, model_name="gan"):
        self.monitor.set_model_name(model_name + "_generator")
        self.monitor.set_model_name(model_name + "_discriminator")
        for epoch in range(epochs):
            gen_train_loss, disc_train_loss = self._train_epoch(epoch, generator, discriminator, gen_optimizer, disc_optimizer, model_name)
            gen_val_loss, disc_val_loss = self._validate_epoch(epoch, generator, discriminator, model_name)
            print(f"Epoch {epoch+1}/{epochs} - Generator Train Loss: {gen_train_loss:.4f}, Discriminator Train Loss: {disc_train_loss:.4f}, Generator Val Loss: {gen_val_loss:.4f}, Discriminator Val Loss: {disc_val_loss:.4f}")
            # Save generator if it has the lowest loss
            
            torch.save(generator.state_dict(), self.model_path + f"/{model_name}_{epoch}_generator.pth")
            torch.save(discriminator.state_dict(), self.model_path + f"/{model_name}_{epoch}_discriminator.pth")
            print(f"Model saved at epoch {epoch+1} with generator loss {gen_val_loss:.4f}")
            
    def _train_epoch(self, epoch, generator, discriminator, gen_optimizer, disc_optimizer, model_name):
        generator.train()
        discriminator.train()
        total_gen_loss = 0.0
        total_disc_loss = 0.0

        for state, action in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}"):
            state, action = state.to(self.device), action.to(self.device)
            batch_size = state.size(0)

            logits = generator(state).detach()
            fake_action = torch.argmax(logits, dim=1, keepdim=True)

            real_labels = torch.ones(batch_size, device=self.device)
            fake_labels = torch.zeros(batch_size, device=self.device)

            disc_optimizer.zero_grad()
            real_score = discriminator(state, action).squeeze()
            fake_score = discriminator(state, fake_action).squeeze()

            disc_loss = self.loss_fn(real_score) - self.loss_fn(fake_score)
            disc_loss.backward()
            disc_optimizer.step()

            logits = generator(state)
            gumbel_softmax = torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=True)
            fake_action = torch.argmax(gumbel_softmax, dim=1, keepdim=True)
            gen_optimizer.zero_grad()
            fake_output = discriminator(state, fake_action).squeeze()
            gen_loss = self.loss_fn(fake_output)
            gen_loss.backward()
            gen_optimizer.step()

            total_disc_loss += disc_loss.item()
            total_gen_loss += gen_loss.item()

            self.monitor.on_train_batch_end(logs={'loss': gen_loss.item(), 'accuracy': 0}, model_name=model_name + "_generator")
            self.monitor.on_train_batch_end(logs={'loss': disc_loss.item(), 'accuracy': 0}, model_name=model_name + "_discriminator")

        avg_gen_loss = total_gen_loss / len(self.train_loader)
        avg_disc_loss = total_disc_loss / len(self.train_loader)
        self.monitor.on_train_epoch_end(logs={'loss': avg_gen_loss, 'accuracy': 0},  model_name=model_name + "_generator")
        self.monitor.on_train_epoch_end(logs={'loss': avg_disc_loss, 'accuracy': 0}, model_name=model_name + "_discriminator")

        return avg_gen_loss, avg_disc_loss
    
    def _validate_epoch(self, epoch, generator, discriminator, model_name):
        generator.eval()
        discriminator.eval()
        total_gen_loss = 0.0
        total_disc_loss = 0.0

        with torch.no_grad():
            for state, action in tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}"):
                state, action = state.to(self.device), action.to(self.device)
                batch_size = state.size(0)

                # === Discriminator Evaluation ===
                logits = generator(state)
                fake_action = torch.argmax(logits, dim=1, keepdim=True)

                real_labels = torch.ones(batch_size, device=self.device)
                fake_labels = torch.zeros(batch_size, device=self.device)

                real_output = discriminator(state, action).squeeze()
                fake_output = discriminator(state, fake_action).squeeze()

                real_loss = self.loss_fn(real_output, real_labels)
                fake_loss = self.loss_fn(fake_output, fake_labels)
                disc_loss = real_loss + fake_loss

                # === Generator Evaluation ===
                # Use updated logits (optionally recompute them)
                fake_output = discriminator(state, fake_action).squeeze()
                gen_loss = self.loss_fn(fake_output, real_labels)

                total_gen_loss += gen_loss.item()
                total_disc_loss += disc_loss.item()

        avg_gen_loss = total_gen_loss / len(self.val_loader)
        avg_disc_loss = total_disc_loss / len(self.val_loader)

        self.monitor.on_val_epoch_end(logs={'loss': avg_gen_loss, 'accuracy': 0}, model_name=model_name + "_generator")
        self.monitor.on_val_epoch_end(logs={'loss': avg_disc_loss, 'accuracy': 0}, model_name=model_name + "_discriminator")

        return avg_gen_loss, avg_disc_loss
