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

    def _gradient_penalty(self, critic, state, real_action_oh, fake_action_oh):
        # Real input
        embedded_state = critic.forward_embedded(state)  # shape: [B, state_embed_dim]
        real_input = torch.cat((embedded_state, real_action_oh), dim=1)  # shape: [B, state_embed_dim + 9]
        fake_input = torch.cat((embedded_state, fake_action_oh), dim=1)  # shape: [B, state_embed_dim + 9]

        # Interpolate between real and fake
        alpha = torch.rand(real_input.size(0), 1).to(self.device)
        alpha = alpha.expand_as(real_input)
        interpolated = (alpha * real_input + (1 - alpha) * fake_input)
        interpolated.requires_grad_(True)

        d_interpolates = critic.forward_layers(interpolated).squeeze(-1)  # shape: [B]

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

    def train(self, epochs: int, generator: ModelDNN, critic: ModelDNN, gen_optimizer, disc_optimizer):
        signal(SIGINT, self._signal_handler)

        for epoch in range(epochs):
            gen_train_loss, disc_train_loss = self._train_epoch(epoch, generator, critic, gen_optimizer, disc_optimizer)
            gen_val_loss, disc_val_loss = self._validate_epoch(epoch, generator, critic)
            print(f"Epoch {epoch+1}/{epochs} - Generator Train Loss: {gen_train_loss:.4f}, Critic Train Loss: {disc_train_loss:.4f}, Generator Val Loss: {gen_val_loss:.4f}, Discriminator Val Loss: {disc_val_loss:.4f}")

            if self.stop_training:
                print("Early stopping triggered.")
                break

        signal(SIGINT, self.original_sigint_handler)

    def _topk_entropy(self, logits, k=3):
        # logits: [B, num_classes]
        probs = F.softmax(logits, dim=-1)             # Convert to probabilities
        topk_probs, _ = torch.topk(probs, k=k, dim=-1) # [B, k]
        entropy = - (topk_probs * torch.log(topk_probs + 1e-12)).sum(dim=-1)  # [B]
        return entropy.mean().item()

    def _train_epoch(self, epoch, generator, critic, gen_optimizer, disc_optimizer):
        generator.train()
        critic.train()
        total_gen_loss = 0.0
        total_disc_loss = 0.0

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
            real_score = critic(state, real_action_oh).squeeze()
            fake_score = critic(state, fake_action_oh).squeeze()

            gp = self._gradient_penalty(critic, state, real_action_oh, fake_action_oh)
            disc_loss = -torch.mean(real_score) + torch.mean(fake_score) + self.lambda_gp * gp

            if prev_disc_loss is not None:
                r_d = abs(disc_loss.item() - prev_disc_loss) / max(abs(prev_disc_loss), 1e-8)
            else:
                r_d = float('inf')
            prev_disc_loss = disc_loss.item()

            # Generator update
            gen_optimizer.zero_grad()
            logits = generator(state)
            fake_action_oh = F.gumbel_softmax(logits, tau=1.0, hard=True)
            fake_output = critic(state, fake_action_oh).squeeze()
            gen_loss = -torch.mean(fake_output)

            if prev_gen_loss is not None:
                r_g = abs(gen_loss.item() - prev_gen_loss) / max(abs(prev_gen_loss), 1e-8)
            else:
                r_g = float('inf')
            prev_gen_loss = gen_loss.item()

            # Dynamic balancing: choose the slower learning network to update
            if self.balance_lambda * r_g < r_d and gen_step * 5 >= disc_step:
                disc_step += 1
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                disc_optimizer.step()
            else:
                gen_step += 1
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                gen_optimizer.step()

            total_disc_loss += disc_loss.item()
            total_gen_loss += gen_loss.item()
            entropy_k3 = self._topk_entropy(logits, k=3)


            self.monitor.on_train_batch_end(model_name=generator.name, key='loss', value=gen_loss.item())
            self.monitor.on_train_batch_end(model_name=critic.name, key='loss', value=disc_loss.item())
            self.monitor.on_train_batch_end(model_name=generator.name, key='score', value=fake_score.mean().item())
            self.monitor.on_train_batch_end(model_name=critic.name, key='score', value=real_score.mean().item())
            self.monitor.on_train_batch_end(model_name=critic.name, key='wasserstein_distance', value=real_score.mean().item() - fake_score.mean().item())
            self.monitor.on_train_batch_end(model_name=critic.name, key='gradient_penalty', value=gp.item())
            self.monitor.on_train_batch_end(model_name=generator.name, key='top3_entropy', value=entropy_k3)


        avg_gen_loss = total_gen_loss / len(self.train_loader)
        avg_disc_loss = total_disc_loss / len(self.train_loader)

        self.monitor.on_train_epoch_end(model_name=generator.name, key='loss', value=avg_gen_loss)
        self.monitor.on_train_epoch_end(model_name=critic.name, key='loss', value=avg_disc_loss)
        self.monitor.on_train_epoch_end(model_name=generator.name, key='step', value=gen_step)
        self.monitor.on_train_epoch_end(model_name=critic.name, key='step', value=disc_step)

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

        self.monitor.on_val_epoch_end(model_name=generator.name, key='loss', value=avg_gen_loss)
        self.monitor.on_val_epoch_end(model_name=discriminator.name, key='loss', value=avg_disc_loss)

        return avg_gen_loss, avg_disc_loss
