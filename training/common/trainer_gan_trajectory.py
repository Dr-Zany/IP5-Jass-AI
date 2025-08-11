import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from signal import signal, getsignal, SIGINT
from typing import Tuple

from .training_monitor import TrainingMonitor
from .model_dnn import ModelDNN
from .jass_env import JassEnv, JassModel


class _EnvRollouts:
    def __init__(self, env_ctor, device: str = 'cpu'):
        self.env_ctor = env_ctor
        self.device = device

    def sample(self, gen_action: ModelDNN, gen_trump_action: ModelDNN, target_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate fake (state, action_slot) pairs by actually playing games with the generator.
        Returns S:[N,72] (long), A_slot:[N,1] (long 0..8).
        """
        collected_S, collected_A = [], []
        policy = JassModel(gen_action, gen_trump_action, device=self.device)
        while sum(x.size(0) for x in collected_A) < target_steps:
            env = self.env_ctor(policy)
            S, A_slot, _A_card = env.play_game()  # updated env returns slot & card; we use slots here
            collected_S.append(S)
            collected_A.append(A_slot)
        S_cat = torch.cat(collected_S, dim=0)[:target_steps]
        A_cat = torch.cat(collected_A, dim=0)[:target_steps]
        return S_cat, A_cat


class TrainerGanTrajectory:
    """WGAN-GP trainer where the critic scores per-candidate card given a state.
    Critic API: critic(X_b36_73) -> [B,36] scores (one per candidate card 0..35).
    Generator API: gen_action(state_b72) -> logits over 9 hand slots (0..8).
    Slot -> card id mapping uses the embedded hand slice in state (no extra inputs required).
    """
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
        self.rollouts = _EnvRollouts(env_ctor=lambda model: JassEnv(model), device=device)

    # ---------------- helpers ----------------
    def _signal_handler(self, signum, frame):
        print(f"Received signal {signum}, stopping training...")
        signal(SIGINT, self.original_sigint_handler)
        self.stop_training = True

    @staticmethod
    def _hand_from_state(state_b72: torch.Tensor) -> torch.Tensor:
        """Extract hand slice from packed state. Layout: 32 history | 3 table | 9 hand | 27 gewisen | 1 trump."""
        return state_b72[:, 35:44].long()  # [B,9] (1..36 or 0)

    @staticmethod
    def _slot_to_card_idx0(state_b72: torch.Tensor, slot_idx_b1: torch.Tensor) -> torch.Tensor:
        """Map slot (0..8) to global card index (0..35) using the hand slice."""
        hand = TrainerGanTrajectory._hand_from_state(state_b72)  # [B,9]
        if slot_idx_b1.dim() == 1:
            slot_idx_b1 = slot_idx_b1.view(-1, 1)
        slot = slot_idx_b1.clamp(min=0, max=8).long()
        cards = hand.gather(1, slot)  # [B,1] -> 1..36 or 0
        # fallback if empty slot (shouldn't happen for legal expert moves, but safe)
        first_nonzero = (hand != 0).long().argmax(dim=1, keepdim=True)
        cards = torch.where(cards == 0, hand.gather(1, first_nonzero), cards)
        return cards.add(-1).clamp(min=0)  # [B,1] -> 0..35

    @staticmethod
    def _pack_candidates(state_b72: torch.Tensor) -> torch.Tensor:
        """Pack candidates for critic: [B,36,73] = [state(72) || action_scalar],
        where action_scalar = candidate_idx / 35.0.
        """
        B = state_b72.size(0)
        device = state_b72.device
        state_f = state_b72.float()
        state_exp = state_f.unsqueeze(1).expand(B, 36, 72)
        cand_idx = torch.arange(36, device=device, dtype=state_f.dtype).unsqueeze(0).expand(B, 36) / 35.0
        action_feat = cand_idx.unsqueeze(-1)
        X = torch.cat([state_exp, action_feat], dim=-1)  # [B,36,73]
        return X

    def _gradient_penalty(self, critic, X_real_b36_73: torch.Tensor, X_fake_b36_73: torch.Tensor, action_idx0_b1: torch.Tensor):
        """WGAN-GP computed on interpolated candidate inputs, gathered on the selected action."""
        B = X_real_b36_73.size(0)
        device = X_real_b36_73.device
        alpha = torch.rand(B, 1, 1, device=device)
        X_interp = alpha * X_real_b36_73 + (1 - alpha) * X_fake_b36_73
        X_interp.requires_grad_(True)
        scores_b36 = critic(X_interp)  # [B,36]
        scores_sel = scores_b36.gather(1, action_idx0_b1.long()).squeeze(1)
        grads = torch.autograd.grad(outputs=scores_sel.sum(), inputs=X_interp, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grads = grads.view(B, -1)
        gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def _topk_entropy(self, logits, k=3):
        probs = F.softmax(logits, dim=-1)
        k = min(k, probs.size(-1))
        topk_probs, _ = torch.topk(probs, k=k, dim=-1)
        return (- (topk_probs * torch.log(topk_probs + 1e-12)).sum(dim=-1)).mean().item()

    # ---------------- train / val ----------------
    def train(self, epochs: int, critic: ModelDNN, gen_action: ModelDNN, gen_trump_action: ModelDNN, gen_optimizer, disc_optimizer, rollout_mul: int = 1):
        signal(SIGINT, self._signal_handler)
        for epoch in range(epochs):
            g_tr, d_tr = self._train_epoch(epoch, critic, gen_action, gen_trump_action, gen_optimizer, disc_optimizer, rollout_mul)
            g_val, d_val = self._validate_epoch(epoch, critic, gen_action)
            print(f"Epoch {epoch+1}/{epochs} - Generator Train Loss: {g_tr:.4f}, Discriminator Train Loss: {d_tr:.4f}, Generator Val Loss: {g_val:.4f}, Discriminator Val Loss: {d_val:.4f}")
            if self.stop_training:
                print("Early stopping triggered.")
                break
        signal(SIGINT, self.original_sigint_handler)

    def _train_epoch(self, epoch, critic: ModelDNN, gen_action: ModelDNN, gen_trump_action: ModelDNN, gen_optimizer, disc_optimizer, rollout_mul: int):
        gen_action.train(); critic.train(); gen_trump_action.train()
        total_gen_loss, total_disc_loss = 0.0, 0.0
        gen_step, disc_step = 0, 0
        prev_gen_loss, prev_disc_loss = None, None
        fake_S_buf, fake_A_buf = None, None

        for state, action_slot in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}"):
            state, action_slot = state.to(self.device), action_slot.to(self.device)  # state [B,72], action_slot [B,1]

            # Expert: slot -> global card idx0 (0..35)
            action_idx0_b1 = self._slot_to_card_idx0(state, action_slot)

            # Rollout fakes
            need = state.size(0)
            if fake_S_buf is None or fake_S_buf.size(0) < need:
                target = need * max(1, rollout_mul)
                S_fake, A_fake_slot = self.rollouts.sample(gen_action, gen_trump_action, target_steps=target)
                fake_S_buf, fake_A_buf = S_fake.to(self.device), A_fake_slot.to(self.device)
            S_fake, A_fake_slot = fake_S_buf[:need], fake_A_buf[:need]
            fake_S_buf, fake_A_buf = fake_S_buf[need:], fake_A_buf[need:]
            fake_action_idx0_b1 = self._slot_to_card_idx0(S_fake, A_fake_slot)

            # Pack candidates
            X_real = self._pack_candidates(state)     # [B,36,73]
            X_fake = self._pack_candidates(S_fake)    # [B,36,73]

            # ---- Discriminator (WGAN-GP) ----
            disc_optimizer.zero_grad()
            real_scores_b36 = critic(X_real)  # [B,36]
            fake_scores_b36 = critic(X_fake)  # [B,36]
            real_score = real_scores_b36.gather(1, action_idx0_b1).squeeze(1)
            fake_score = fake_scores_b36.gather(1, fake_action_idx0_b1).squeeze(1)
            gp = self._gradient_penalty(critic, X_real, X_fake, action_idx0_b1)
            disc_loss = -torch.mean(real_score) + torch.mean(fake_score) + self.lambda_gp * gp

            r_d = abs(disc_loss.item() - prev_disc_loss) / max(abs(prev_disc_loss), 1e-8) if prev_disc_loss is not None else float('inf')
            prev_disc_loss = disc_loss.item()

            # ---- Generator (on expert states for stability) ----
            gen_optimizer.zero_grad()
            logits9 = gen_action(state)                                   # [B,9]
            fake_slot_oh = F.gumbel_softmax(logits9, tau=1.0, hard=True)  # [B,9]
            fake_slot_idx_b1 = fake_slot_oh.argmax(dim=1, keepdim=True)   # [B,1]
            fake_card_idx0_b1 = self._slot_to_card_idx0(state, fake_slot_idx_b1)

            X_for_gen = self._pack_candidates(state)
            scores_b36 = critic(X_for_gen)
            fake_score_gen = scores_b36.gather(1, fake_card_idx0_b1).squeeze(1)
            gen_loss = -torch.mean(fake_score_gen)

            r_g = abs(gen_loss.item() - prev_gen_loss) / max(abs(prev_gen_loss), 1e-8) if prev_gen_loss is not None else float('inf')
            prev_gen_loss = gen_loss.item()

            # dynamic balancing
            if self.balance_lambda * r_g < r_d and gen_step * 5 >= disc_step:
                disc_step += 1
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
                disc_optimizer.step()
            else:
                gen_step += 1
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(gen_action.parameters(), max_norm=1.0)
                gen_optimizer.step()

            total_disc_loss += disc_loss.item()
            total_gen_loss += gen_loss.item()
            entropy_k3 = self._topk_entropy(logits9, k=3)

            # monitoring
            self.monitor.on_train_batch_end(model_name=getattr(gen_action, 'name', 'generator'), key='loss', value=gen_loss.item())
            self.monitor.on_train_batch_end(model_name=getattr(critic, 'name', 'critic'), key='loss', value=disc_loss.item())
            self.monitor.on_train_batch_end(model_name=getattr(gen_action, 'name', 'generator'), key='top3_entropy', value=entropy_k3)
            self.monitor.on_train_batch_end(model_name=getattr(critic, 'name', 'critic'), key='wasserstein_distance', value=real_score.mean().item() - fake_score.mean().item())
            self.monitor.on_train_batch_end(model_name=getattr(critic, 'name', 'critic'), key='gradient_penalty', value=gp.item())

        avg_gen_loss = total_gen_loss / len(self.train_loader)
        avg_disc_loss = total_disc_loss / len(self.train_loader)

        self.monitor.on_train_epoch_end(model_name=getattr(gen_action, 'name', 'generator'), key='loss', value=avg_gen_loss)
        self.monitor.on_train_epoch_end(model_name=getattr(critic, 'name', 'critic'), key='loss', value=avg_disc_loss)
        self.monitor.on_train_epoch_end(model_name=getattr(gen_action, 'name', 'generator'), key='step', value=gen_step)
        self.monitor.on_train_epoch_end(model_name=getattr(critic, 'name', 'critic'), key='step', value=disc_step)

        print(f"Generator steps: {gen_step}, Discriminator steps: {disc_step}")
        return avg_gen_loss, avg_disc_loss

    def _validate_epoch(self, epoch, critic: ModelDNN, gen_action: ModelDNN):
        gen_action.eval(); critic.eval()
        total_gen_loss, total_disc_loss = 0.0, 0.0
        with torch.no_grad():
            for state, action_slot in tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}"):
                state, action_slot = state.to(self.device), action_slot.to(self.device)
                action_idx0_b1 = self._slot_to_card_idx0(state, action_slot)

                X = self._pack_candidates(state)
                scores_b36 = critic(X)
                real_score = scores_b36.gather(1, action_idx0_b1).squeeze(1)

                logits9 = gen_action(state)
                fake_slot_oh = F.gumbel_softmax(logits9, tau=1.0, hard=True)
                fake_slot_idx_b1 = fake_slot_oh.argmax(dim=1, keepdim=True)
                fake_card_idx0_b1 = self._slot_to_card_idx0(state, fake_slot_idx_b1)
                fake_score = scores_b36.gather(1, fake_card_idx0_b1).squeeze(1)

                disc_loss = -torch.mean(real_score) + torch.mean(fake_score)
                gen_loss = -torch.mean(fake_score)

                total_gen_loss += gen_loss.item()
                total_disc_loss += disc_loss.item()

        avg_gen_loss = total_gen_loss / len(self.val_loader)
        avg_disc_loss = total_disc_loss / len(self.val_loader)

        self.monitor.on_val_epoch_end(model_name=getattr(gen_action, 'name', 'generator'), key='loss', value=avg_gen_loss)
        self.monitor.on_val_epoch_end(model_name=getattr(critic, 'name', 'critic'), key='loss', value=avg_disc_loss)
        return avg_gen_loss, avg_disc_loss
