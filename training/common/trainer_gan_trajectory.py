import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from .training_monitor import TrainingMonitor
from .model_dnn import ModelDNN
from signal import signal, getsignal, SIGINT
from .jass_env import JassModel, JassEnv
from itertools import islice


# wgan-gp trainer for GANs with dynamic balancing
class TrainerGanTrajectory:
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
        self.jass_env = JassEnv()  # Placeholder for JassModel

    def _signal_handler(self, signum, frame):
        print(f"Received signal {signum}, stopping training...")
        signal(SIGINT, self.original_sigint_handler)
        self.stop_training = True

    def _gradient_penalty(self, critic, real_state, real_action_oh, fake_state, fake_action_oh):
        """
        real_state:     [B, 36, 72]
        real_action_oh: [B, 36, 9]
        fake_state:     [B, 36, 72]
        fake_action_oh: [B, 36, 9]
        """
        device = self.device

        real_state     = real_state.to(device)
        fake_state     = fake_state.to(device)
        real_action_oh = real_action_oh.to(device).float()
        fake_action_oh = fake_action_oh.to(device).float()

        B = real_state.size(0)

        # same alpha for state & action
        alpha = torch.rand(B, 1, 1, device=device)
        interp_state  = (alpha * real_state + (1 - alpha) * fake_state).requires_grad_(True)
        interp_action = (alpha * real_action_oh + (1 - alpha) * fake_action_oh).requires_grad_(True)

        # critic outputs per card; could be [B, 36] or [B, 36, 1]
        d_interpolates = critic(interp_state, interp_action)
        if d_interpolates.dim() == 3 and d_interpolates.size(-1) == 1:
            d_interpolates = d_interpolates.squeeze(-1)  # [B, 36]

        grad_outputs = torch.ones_like(d_interpolates, device=device)

        g_state, g_action = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=[interp_state, interp_action],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )

        # Flatten per-sample (use reshape to handle non-contiguous grads)
        g_state  = g_state.reshape(B, -1)
        g_action = g_action.reshape(B, -1)
        g_total  = torch.cat([g_state, g_action], dim=1)

        gradient_penalty = ((g_total.norm(2, dim=1) - 1.0) ** 2).mean()
        return gradient_penalty



    def train(self, epochs: int, critic: ModelDNN, generator: ModelDNN, trumper: ModelDNN, gen_optimizer, disc_optimizer):
        signal(SIGINT, self._signal_handler)

        for epoch in range(epochs):
            gen_train_loss, disc_train_loss = self._train_epoch(epoch, critic, generator, trumper, gen_optimizer, disc_optimizer)
            gen_val_loss, disc_val_loss = self._validate_epoch(epoch, generator, trumper, critic)
            print(f"Epoch {epoch+1}/{epochs} - Generator Train Loss: {gen_train_loss:.4f}, Discriminator Train Loss: {disc_train_loss:.4f}, Generator Val Loss: {gen_val_loss:.4f}, Discriminator Val Loss: {disc_val_loss:.4f}")

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
    

    def _train_epoch(self, epoch, critic, generator, trumper, gen_optimizer, disc_optimizer):
        generator.train()
        critic.train()
        total_gen_loss = 0.0
        total_disc_loss = 0.0

        gen_step = 0
        disc_step = 0

        prev_gen_loss = None
        prev_disc_loss = None

        jass_model = JassModel(model_jass=generator, model_trump=trumper, device=self.device)

        for state, action in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}"):
            real_states, real_actions = state.to(self.device), action.to(self.device)
            real_actions_oh = F.one_hot(real_actions.squeeze(-1), num_classes=9).float()

            fake_states, fake_actions_oh = self.jass_env.play_game(model=jass_model, B=real_states.size(0))
            fake_states = fake_states.to(self.device)
            fake_actions_oh = fake_actions_oh.to(self.device)
            disc_optimizer.zero_grad()
            gen_optimizer.zero_grad()

            real_score = critic(real_states, real_actions_oh).squeeze()
            fake_score = critic(fake_states.detach(), fake_actions_oh.detach()).squeeze()

            gp = self._gradient_penalty(
                critic,
                real_states,
                real_actions_oh,
                fake_states.detach(),
                fake_actions_oh.detach(),
            )
            disc_loss = -torch.mean(real_score) + torch.mean(fake_score) + self.lambda_gp * gp

            if prev_disc_loss is not None:
                r_d = abs(disc_loss.item() - prev_disc_loss) / max(abs(prev_disc_loss), 1e-8)
            else:
                r_d = float('inf')
            prev_disc_loss = disc_loss.item()

            fake_output = critic(fake_states, fake_actions_oh).squeeze()
            gen_loss = -torch.mean(fake_output)

            if prev_gen_loss is not None:
                r_g = abs(gen_loss.item() - prev_gen_loss) / max(abs(prev_gen_loss), 1e-8)
            else:
                r_g = float('inf')
            prev_gen_loss = gen_loss.item()

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
            # entropy_k3 = self._topk_entropy(fake_actions_oh, k=3)


            self.monitor.on_train_batch_end(model_name=generator.name, key='loss', value=gen_loss.item())
            self.monitor.on_train_batch_end(model_name=critic.name, key='loss', value=disc_loss.item())
            self.monitor.on_train_batch_end(model_name=generator.name, key='score', value=fake_score.mean().item())
            self.monitor.on_train_batch_end(model_name=critic.name, key='score', value=real_score.mean().item())
            self.monitor.on_train_batch_end(model_name=critic.name, key='wasserstein_distance', value=real_score.mean().item() - fake_score.mean().item())
            self.monitor.on_train_batch_end(model_name=critic.name, key='gradient_penalty', value=gp.item())
            self.monitor.on_train_batch_end(model_name=generator.name, key='top3_entropy', value=0)


        avg_gen_loss = total_gen_loss / len(self.train_loader)
        avg_disc_loss = total_disc_loss / len(self.train_loader)

        self.monitor.on_train_epoch_end(model_name=generator.name, key='loss', value=avg_gen_loss)
        self.monitor.on_train_epoch_end(model_name=critic.name, key='loss', value=avg_disc_loss)
        self.monitor.on_train_epoch_end(model_name=generator.name, key='step', value=gen_step)
        self.monitor.on_train_epoch_end(model_name=critic.name, key='step', value=disc_step)

        print(f"Generator steps: {gen_step}, Discriminator steps: {disc_step}")

        return avg_gen_loss, avg_disc_loss

    def _validate_epoch(self, epoch, generator, trumper, critic):
        generator.eval()
        critic.eval()
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        total_disc_acc = 0.0

        jass_model = JassModel(model_jass=generator, model_trump=trumper, device=self.device)
        with torch.no_grad():
            for state, action in tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}"):
                real_states, real_actions = state.to(self.device), action.to(self.device)
                real_actions_oh = F.one_hot(real_actions.squeeze(-1), num_classes=9).float()

                fake_states, fake_actions_oh = self.jass_env.play_game(model=jass_model, B=real_states.size(0))
                fake_states = fake_states.to(self.device)
                fake_actions_oh = fake_actions_oh.to(self.device)
                real_score = critic(real_states, real_actions_oh).squeeze()
                fake_score = critic(fake_states, fake_actions_oh).squeeze()

                disc_loss = -torch.mean(real_score) + torch.mean(fake_score)
                gen_loss = -torch.mean(fake_score)

                real_pred = (real_score > 0).float()
                fake_pred = (fake_score < 0).float()
                disc_acc = 0.5 * (real_pred.mean().item() + fake_pred.mean().item())

                total_gen_loss += gen_loss.item()
                total_disc_loss += disc_loss.item()
                total_disc_acc += disc_acc

        avg_gen_loss = total_gen_loss / len(self.val_loader)
        avg_disc_loss = total_disc_loss / len(self.val_loader)

        self.monitor.on_val_epoch_end(model_name=generator.name, key='loss', value=avg_gen_loss)
        self.monitor.on_val_epoch_end(model_name=critic.name, key='loss', value=avg_disc_loss)

        return avg_gen_loss, avg_disc_loss
