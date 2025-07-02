import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from .training_monitor import TrainingMonitor
from .model_dnn import ModelDNN
from signal import signal, getsignal, SIGINT

# wgan-gp trainer for GANs with dynamic balancing
class TrainerGan:
    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, model_path: str, lambda_gp=10.0, balance_lambda=1.0, device='cpu'):
        self.model_path = model_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lambda_gp = lambda_gp
        self.balance_lambda = balance_lambda
        self.device = device
        self.original_sigint_handler = getsignal(SIGINT)
        self.stop_training = False
        self.monitor = TrainingMonitor()

    def _signal_handler(self, signum, frame):
        print(f"Received signal {signum}, stopping training...")
        signal(SIGINT, self.original_sigint_handler)
        self.stop_training = True

    def _gradient_penalty(self, discriminator, state, real_action_oh, fake_action_oh):
        # Real input
        embedded_state = discriminator.forward_embedded(state)  # shape: [B, state_embed_dim]
        real_input = torch.cat((embedded_state, real_action_oh), dim=1)  # shape: [B, state_embed_dim + 9]
        fake_input = torch.cat((embedded_state, fake_action_oh), dim=1)  # shape: [B, state_embed_dim + 9]

        # Interpolate between real and fake
        alpha = torch.rand(real_input.size(0), 1).to(self.device)
        alpha = alpha.expand_as(real_input)
        interpolated = (alpha * real_input + (1 - alpha) * fake_input)
        interpolated.requires_grad_(True)

        d_interpolates = discriminator.forward_layers(interpolated).squeeze(-1)  # shape: [B]

        gradients = torch.autograd.grad(
            outputs=d_interpolates.sum(),
            inputs=interpolated,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  # shape: [B, state_embed_dim + 9]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()  # scalar
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

        gen_step = 0
        disc_step = 0

        prev_gen_loss = None
        prev_disc_loss = None

        for state, action in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}"):
            state, action = state.to(self.device), action.to(self.device)
            real_action_oh = F.one_hot(action.squeeze(dim=1), num_classes=9).float()

            logits = generator(state).detach()
            fake_action_oh = F.gumbel_softmax(logits, tau=1.0, hard=True)

            # Discriminator update
            disc_optimizer.zero_grad()
            real_score = discriminator(state, real_action_oh).squeeze()
            fake_score = discriminator(state, fake_action_oh).squeeze()

            gp = self._gradient_penalty(discriminator, state, real_action_oh, fake_action_oh)
            disc_loss = -torch.mean(real_score) + torch.mean(fake_score) + self.lambda_gp * gp

            real_pred = (real_score > 0).float()
            fake_pred = (fake_score < 0).float()
            disc_acc = 0.5 * (real_pred.mean().item() + fake_pred.mean().item())

            if prev_disc_loss is not None:
                r_d = abs(disc_loss.item() - prev_disc_loss) / max(abs(prev_disc_loss), 1e-8)
            else:
                r_d = float('inf')
            prev_disc_loss = disc_loss.item()

            # Generator update
            gen_optimizer.zero_grad()
            logits = generator(state)
            fake_action_oh = F.gumbel_softmax(logits, tau=1.0, hard=True)
            fake_output = discriminator(state, fake_action_oh).squeeze()
            gen_loss = -torch.mean(fake_output)

            # Add behavioral cloning loss as hint
            # bc_loss = F.cross_entropy(logits, action.squeeze(dim=1))
            # gen_loss += 0.2 * bc_loss

            if prev_gen_loss is not None:
                r_g = abs(gen_loss.item() - prev_gen_loss) / max(abs(prev_gen_loss), 1e-8)
            else:
                r_g = float('inf')
            prev_gen_loss = gen_loss.item()

            # Dynamic balancing: choose the slower learning network to update
            if self.balance_lambda * r_g < r_d:
                disc_step += 1
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                disc_optimizer.step()
            else:
                gen_step += 1
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                gen_optimizer.step()

            gen_acc = logits.argmax(dim=1).eq(action).float().mean().item()

            total_disc_loss += disc_loss.item()
            total_disc_acc += disc_acc
            total_gen_loss += gen_loss.item()
            total_gen_acc += gen_acc

            self.monitor.on_train_batch_end(logs={'loss': gen_loss.item(), 'accuracy': gen_acc}, model_name=generator.name)
            self.monitor.on_train_batch_end(logs={'loss': disc_loss.item(), 'accuracy': disc_acc}, model_name=discriminator.name)

        avg_gen_loss = total_gen_loss / len(self.train_loader)
        avg_disc_loss = total_disc_loss / len(self.train_loader)
        avg_gen_acc = total_gen_acc / len(self.train_loader)
        avg_disc_acc = total_disc_acc / len(self.train_loader)
        self.monitor.on_train_epoch_end(logs={'loss': avg_gen_loss, 'accuracy': avg_gen_acc}, model_name=generator.name)
        self.monitor.on_train_epoch_end(logs={'loss': avg_disc_loss, 'accuracy': avg_disc_acc}, model_name=discriminator.name)

        print(f"Generator steps: {gen_step}, Discriminator steps: {disc_step}")

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
                real_action_oh = F.one_hot(action.squeeze(dim=1), num_classes=9).float()

                logits = generator(state)
                fake_action_oh = F.gumbel_softmax(logits, tau=1.0, hard=True)

                real_score = discriminator(state, real_action_oh).squeeze()
                fake_score = discriminator(state, fake_action_oh).squeeze()

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
