import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from .training_monitor import TrainingMonitor
from .model_dnn import ModelDNN
from signal import signal, getsignal, SIGINT

# wgan-gp trainer for GANs
class TrainerGan:
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, model_path: str, n_critic=1, n_gen=1, lambda_gp=10.0, device='cpu'):
        self.model_path = model_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.n_critic = n_critic
        self.n_gen = n_gen
        self.lambda_gp = lambda_gp
        self.device = device
        self.original_sigint_handler = getsignal(SIGINT)
        self.stop_training = False
        self.monitor = TrainingMonitor()

    def _signal_handler(self, signum, frame):
        print(f"Received signal {signum}, stopping training...")
        signal(SIGINT, self.original_sigint_handler)
        self.stop_training = True

    def _gradient_penalty(self, discriminator, real_state, real_action):
        real_input = torch.cat((real_state, real_action), dim=1)
        other_state = real_state[torch.randperm(real_state.size(0))]
        other_input = torch.cat((other_state, real_action), dim=1)

        embed_real = discriminator.embedding(real_input)
        embed_other = discriminator.embedding(other_input)

        alpha = torch.rand(embed_real.size(0), 1, 1, device=self.device)
        alpha = alpha.expand_as(embed_real)
        interpolated = (alpha * embed_real + (1 - alpha) * embed_other).detach().requires_grad_(True)

        interpolated_flat = interpolated.view(interpolated.size(0), -1)
        d_interpolates = discriminator.forward_layers(interpolated_flat).squeeze()

        gradients = torch.autograd.grad(
            outputs=d_interpolates.sum(),
            inputs=interpolated,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train(self, epochs: int, generator: ModelDNN, discriminator: ModelDNN, gen_optimizer, disc_optimizer):
        signal(SIGINT, self._signal_handler)
        self.monitor.set_model_name(generator.name)
        self.monitor.set_model_name(discriminator.name)

        for epoch in range(epochs):
            gen_train_loss, disc_train_loss = self._train_epoch(epoch, generator, discriminator, gen_optimizer, disc_optimizer)
            gen_val_loss, disc_val_loss = self._validate_epoch(epoch, generator, discriminator)
            print(f"Epoch {epoch+1}/{epochs} - Generator Train Loss: {gen_train_loss:.4f}, Discriminator Train Loss: {disc_train_loss:.4f}, Generator Val Loss: {gen_val_loss:.4f}, Discriminator Val Loss: {disc_val_loss:.4f}")

            if self.stop_training:
                print("Early stopping triggered.")
                break

        signal(SIGINT, self.original_sigint_handler)

    def _train_epoch(self, epoch, generator, discriminator, gen_optimizer, disc_optimizer):
        generator.train()
        discriminator.train()
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        total_gen_acc = 0.0
        total_disc_acc = 0.0

        for state, action in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}"):
            state, action = state.to(self.device), action.to(self.device)

            logits = generator(state).detach()
            fake_action = torch.argmax(logits, dim=1, keepdim=True)

            disc_loss = None
            disc_acc = None

            for _ in range(self.n_critic):
                disc_optimizer.zero_grad()
                real_score = discriminator(state, action).squeeze()
                fake_score = discriminator(state, fake_action).squeeze()

                gp = self._gradient_penalty(discriminator, state, action)
                disc_loss = -torch.mean(real_score) + torch.mean(fake_score) + self.lambda_gp * gp

                # Loss feedback control
                if disc_loss.item() > 0.5:
                    disc_loss.backward()
                    disc_optimizer.step()

                real_pred = (real_score > 0).float()
                fake_pred = (fake_score < 0).float()
                disc_acc = 0.5 * (real_pred.mean().item() + fake_pred.mean().item())
                total_disc_acc += disc_acc

            for _ in range(self.n_gen):
                gen_optimizer.zero_grad()
                logits = generator(state)
                fake_action = torch.argmax(logits, dim=1, keepdim=True)
                fake_output = discriminator(state, fake_action).squeeze()
                gen_loss = -torch.mean(fake_output)
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                gen_optimizer.step()

                gen_acc = logits.argmax(dim=1).eq(action).float().mean().item()

            total_disc_loss += disc_loss.item() if disc_loss is not None else 0.0
            total_gen_loss += gen_loss.item()
            total_gen_acc += gen_acc

            self.monitor.on_train_batch_end(logs={'loss': gen_loss.item(), 'accuracy': gen_acc}, model_name=generator.name)
            if disc_loss is not None:
                self.monitor.on_train_batch_end(logs={'loss': disc_loss.item(), 'accuracy': disc_acc}, model_name=discriminator.name)

        avg_gen_loss = total_gen_loss / len(self.train_loader)
        avg_disc_loss = total_disc_loss / len(self.train_loader)
        avg_gen_acc = total_gen_acc / len(self.train_loader)
        avg_disc_acc = total_disc_acc / len(self.train_loader)
        self.monitor.on_train_epoch_end(logs={'loss': avg_gen_loss, 'accuracy': avg_gen_acc}, model_name=generator.name)
        self.monitor.on_train_epoch_end(logs={'loss': avg_disc_loss, 'accuracy': avg_disc_acc}, model_name=discriminator.name)

        return avg_gen_loss, avg_disc_loss

    def _validate_epoch(self, epoch, generator, discriminator):
        generator.eval()
        discriminator.eval()
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        total_gen_acc = 0.0
        total_disc_acc = 0.0

        with torch.no_grad():
            for state, action in tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}"):
                state, action = state.to(self.device), action.to(self.device)

                logits = generator(state)
                fake_action = torch.argmax(logits, dim=1, keepdim=True)

                real_score = discriminator(state, action).squeeze()
                fake_score = discriminator(state, fake_action).squeeze()


                disc_loss = -torch.mean(real_score) + torch.mean(fake_score)
                gen_loss = -torch.mean(fake_score)

                real_pred = (real_score > 0).float()
                fake_pred = (fake_score < 0).float()
                disc_acc = 0.5 * (real_pred.mean().item() + fake_pred.mean().item())
                gen_acc = logits.argmax(dim=1).eq(action).float().mean().item()

                total_gen_loss += gen_loss.item()
                total_disc_loss += disc_loss.item()
                total_gen_acc += gen_acc
                total_disc_acc += disc_acc

        avg_gen_loss = total_gen_loss / len(self.val_loader)
        avg_disc_loss = total_disc_loss / len(self.val_loader)
        avg_gen_acc = total_gen_acc / len(self.val_loader)
        avg_disc_acc = total_disc_acc / len(self.val_loader)

        self.monitor.on_val_epoch_end(logs={'loss': avg_gen_loss, 'accuracy': avg_gen_acc}, model_name=generator.name)
        self.monitor.on_val_epoch_end(logs={'loss': avg_disc_loss, 'accuracy': avg_disc_acc}, model_name=discriminator.name)

        return avg_gen_loss, avg_disc_loss
