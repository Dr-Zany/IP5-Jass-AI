# train_gail_with_progress.py

import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ─── hyperparameters & paths ─────────────────────────────────────────
DATASET_PATH = "playing.hdf5"
MODEL_DIR = "./models"
BATCH_SIZE_EXP = 256       # expert minibatch size
DISCRIM_ITERS = 5          # how many discriminator updates per epoch
POLICY_LR = 1e-4
DISCRIM_LR = 1e-4
NUM_EPOCHS = 200
SEED = 42
NUM_ONPOL_EP = 16          # number of on-policy games per epoch

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── environment state constants ──────────────────────────────────────
NUM_TRUMPS = 1
NUM_CARDS_HAND = 9
NUM_CARDS_TABLE = 4
NUM_CARDS_HISTORY = 32
NUM_CARDS_SHOWN = 27
NUM_STATE = NUM_TRUMPS + NUM_CARDS_HAND + \
    NUM_CARDS_TABLE + NUM_CARDS_HISTORY + NUM_CARDS_SHOWN
# ──────────────────────────────────────────────────────────────────────


# ─── expert transition dataset ────────────────────────────────────────
class TransitionDataset(Dataset):
    def __init__(self, h5_path):
        self.h5 = h5py.File(h5_path, "r")
        self.episodes = list(self.h5.keys())
        lengths = [self.h5[e]["state"].shape[0] for e in self.episodes]
        self.cumlen = np.cumsum([0] + lengths)
        self.total = int(self.cumlen[-1])

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        ep_idx = np.searchsorted(self.cumlen, idx, side="right") - 1
        within = int(idx - self.cumlen[ep_idx])
        grp = self.h5[self.episodes[ep_idx]]
        s = torch.from_numpy(grp["state"][within]).float()    # [NUM_STATE]
        # scalar in [0..8]
        a = int(grp["action"][within, 0])
        return s, a


# ─── models ───────────────────────────────────────────────────────────
class PolicyMLP(nn.Module):
    def __init__(self, state_dim=NUM_STATE, hidden_sizes=[512, 256, 128], num_actions=9):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, num_actions)

    def forward(self, x):
        # x: [batch, NUM_STATE]
        x = self.body(x)
        logits = self.head(x)
        return F.log_softmax(logits, dim=-1)


class DiscriminatorMLP(nn.Module):
    def __init__(self, state_dim=NUM_STATE, action_dim=9, hidden_sizes=[256, 256]):
        super().__init__()
        input_dim = state_dim + action_dim
        dims = [input_dim] + hidden_sizes + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, s, a):
        # s: [batch, NUM_STATE] float, a: [batch] long
        a_oh = F.one_hot(a, num_classes=9).float()     # [batch,9]
        x = torch.cat([s, a_oh], dim=1)                # [batch, NUM_STATE+9]
        return torch.sigmoid(self.net(x)).squeeze(1)   # [batch]


# ─── main training loop ───────────────────────────────────────────────
def train():
    # 1) Expert loader
    exp_ds = TransitionDataset(DATASET_PATH)
    exp_loader = DataLoader(
        exp_ds, batch_size=BATCH_SIZE_EXP, shuffle=True, drop_last=True
    )
    exp_iter = iter(exp_loader)

    # 2) Instantiate models
    policy = PolicyMLP(state_dim=NUM_STATE).to(device)
    disc = DiscriminatorMLP(state_dim=NUM_STATE, action_dim=9).to(device)

    opt_pol = optim.Adam(policy.parameters(), lr=POLICY_LR)
    opt_dis = optim.Adam(disc.parameters(),   lr=DISCRIM_LR)

    # 3) Create environment
    from jass_gym.gym_env import SchieberEnvFlat
    env = SchieberEnvFlat(point_limit=2500, seed=SEED)

    # 4) Loop over epochs, with tqdm progress bar
    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs"):
        # ——— 1) collect on-policy transitions —————————————————————
        on_states, on_actions = [], []
        for _ in range(NUM_ONPOL_EP):
            s = env.reset()  # numpy array [NUM_STATE]
            done = False
            while not done:
                s_t = torch.from_numpy(s).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    logp = policy(s_t)                # [1,9]
                a = logp.exp().multinomial(1).item()  # sample action
                s2, _, done, _ = env.step(a)
                on_states.append(s)
                on_actions.append(a)
                s = s2

        on_states = torch.tensor(on_states, dtype=torch.float32).to(
            device)  # [N_on, NUM_STATE]
        on_actions = torch.tensor(
            on_actions, dtype=torch.long).to(device)   # [N_on]

        # ——— 2) update discriminator —————————————————————————
        # we’ll accumulate a running average of discriminator loss
        d_loss_avg = 0.0
        for _ in range(DISCRIM_ITERS):
            try:
                exp_s, exp_a = next(exp_iter)
            except StopIteration:
                exp_iter = iter(exp_loader)
                exp_s, exp_a = next(exp_iter)

            exp_s = exp_s.to(device)                    # [B,NUM_STATE]
            exp_a = exp_a.to(device)                    # [B]

            D_exp = disc(exp_s, exp_a)                  # [B]
            D_gen = disc(on_states, on_actions)         # [N_on]

            # Discriminator loss: - E[log D(exp)] - E[log(1 - D(gen))]
            loss_d = - (torch.log(D_exp + 1e-8).mean() +
                        torch.log(1.0 - D_gen + 1e-8).mean())

            opt_dis.zero_grad()
            loss_d.backward()
            opt_dis.step()

            d_loss_avg += loss_d.item()

        d_loss_avg /= DISCRIM_ITERS

        # ——— 3) compute imitation rewards & update policy —————————
        with torch.no_grad():
            D_gen = disc(on_states, on_actions)         # [N_on]
            rewards = - torch.log(1.0 - D_gen + 1e-8)    # [N_on]

        logps = policy(on_states)                       # [N_on,9]
        logp_sa = logps.gather(1, on_actions.unsqueeze(1)).squeeze(1)  # [N_on]
        loss_p = - (rewards * logp_sa).mean()

        opt_pol.zero_grad()
        loss_p.backward()
        opt_pol.step()

        # ——— 4) logging & checkpoint —————————————————————————————————
        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | D_loss={d_loss_avg:.4f} | P_loss={loss_p.item():.4f} | "
                f"Rollouts={on_states.size(0)}"
            )
            torch.save(policy.state_dict(), os.path.join(
                MODEL_DIR, f"policy_ep{epoch}.pt"))
            torch.save(disc.state_dict(),   os.path.join(
                MODEL_DIR, f"disc_ep{epoch}.pt"))


if __name__ == "__main__":
    train()
