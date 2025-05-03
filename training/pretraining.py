import os
import numpy as np
import h5py
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import types, rollout
from imitation.util import logger as imit_logger
from imitation.algorithms.bc import BC

# --- Feature Extractor ---
class JassFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for Jass observations using an embedding layer."""
    def __init__(self, observation_space, embedding_dim=16):
        n_cards = 37  # 36 cards + 1 for 'none'
        features_dim = int(np.prod(observation_space.shape)) * embedding_dim
        super().__init__(observation_space, features_dim)
        self.embedding = nn.Embedding(n_cards, embedding_dim)

    def forward(self, observations):
        x = observations.long()
        embedded = self.embedding(x)
        return embedded.view(x.shape[0], -1)

# --- Environment ---
class JassEnv(gym.Env):
    """Custom Gym environment for Jass game episodes loaded from HDF5."""
    metadata = {"render_modes": [], "render_fps": 1}

    def __init__(self, hdf5_path, groups, rng=None):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.hdf5 = h5py.File(hdf5_path, 'r')
        self.groups = list(groups)
        self.rng = rng or np.random.default_rng()
        # Discrete card IDs for 72 positions
        self.observation_space = spaces.Box(low=0, high=36, shape=(72,), dtype=np.uint8)
        self.action_space = spaces.Discrete(9)
        self.last_obs = None
        self._load_new_episode()

    def _load_new_episode(self):
        if not self.groups:
            raise RuntimeError("No groups available to load an episode.")
        self.current_group_name = self.rng.choice(self.groups)
        grp = self.hdf5[self.current_group_name]
        self.current_group_data = {
            'states': grp['state'][:].astype(np.uint8),
            'actions': grp['action'][:].astype(np.uint8)
        }
        self.total_steps_in_group = len(self.current_group_data['states'])
        if self.total_steps_in_group == 0:
            self.groups.remove(self.current_group_name)
            return self._load_new_episode()
        self.current_index_in_group = 0
        self.last_obs = self.current_group_data['states'][0]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._load_new_episode()
        obs = self.current_group_data['states'][0]
        self.last_obs = obs
        return obs, {}

    def step(self, action):
        if self.current_index_in_group >= self.total_steps_in_group:
            raise RuntimeError("Step called after episode ended.")
        expert_action = self.current_group_data['actions'][self.current_index_in_group]
        reward = 1.0 if action == expert_action else 0.0
        self.current_index_in_group += 1
        terminated = self.current_index_in_group >= self.total_steps_in_group
        if not terminated:
            obs = self.current_group_data['states'][self.current_index_in_group]
            self.last_obs = obs
        else:
            obs = self.last_obs
        return obs, reward, terminated, False, {'expert_action': expert_action}

    def close(self):
        if hasattr(self, 'hdf5') and self.hdf5:
            self.hdf5.close()
            self.hdf5 = None

    def __del__(self):
        self.close()

# --- Data Loader ---
def load_transitions(hdf5_path, groups):
    all_obs, all_acts, all_infos, all_dones = [], [], [], []
    with h5py.File(hdf5_path, 'r') as f:
        for group_name in groups:
            try:
                grp = f[group_name]
                states = grp['state'][:].astype(np.uint8)
                acts = grp['action'][:].astype(np.uint8)
                if len(states) != len(acts) or len(states) == 0:
                    continue
                all_obs.append(states)
                all_acts.append(acts)
                all_infos.extend([{}] * len(states))
                dones = np.zeros(len(states), dtype=bool)
                dones[-1] = True
                all_dones.append(dones)
            except Exception:
                continue
    if not all_obs:
        raise ValueError("No valid transitions loaded.")
    obs_arr = np.concatenate(all_obs)
    acts_arr = np.concatenate(all_acts)
    dones_arr = np.concatenate(all_dones)
    next_obs_arr = np.roll(obs_arr, -1, axis=0)
    next_obs_arr[dones_arr] = obs_arr[dones_arr]
    return types.Transitions(
        obs=obs_arr,
        acts=acts_arr,
        infos=np.array(all_infos),
        next_obs=next_obs_arr,
        dones=dones_arr
    )

# --- Data Splitter ---
def split_groups(hdf5_path, test_fraction=0.2, rng=None):
    rng = rng or np.random.default_rng()
    with h5py.File(hdf5_path, 'r') as f:
        all_groups = list(f.keys())
        total_states = f.attrs.get('total_states_saved', sum(len(f[g]['state']) for g in all_groups))
    target_test = total_states * test_fraction
    rng.shuffle(all_groups)
    test_g, train_g, count = [], [], 0
    with h5py.File(hdf5_path, 'r') as f:
        for g in all_groups:
            n = f[g].attrs.get('total_states_saved', len(f[g]['state']))
            (test_g if count < target_test else train_g).append(g)
            count += n
    return train_g, test_g

# --- Main Function ---
def main():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    hdf5_path = '../log_parser/jass_dataset.hdf5'
    log_dir = './logs_bc/'
    rng_seed = 42

    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    rng = np.random.default_rng(rng_seed)

    os.makedirs(log_dir, exist_ok=True)
    logger = imit_logger.configure(folder=log_dir, format_strs=["stdout", "log", "csv", "tensorboard"])

    print("Splitting data...")
    train_groups, test_groups = split_groups(hdf5_path, 0.2, rng)
    print(f"Train episodes: {len(train_groups)}, Test episodes: {len(test_groups)}")

    print("Loading transitions...")
    train_trans = load_transitions(hdf5_path, train_groups)
    test_trans = load_transitions(hdf5_path, test_groups)

    print("Setting up evaluation env...")
    num_envs = 8
    def make_env(): return Monitor(JassEnv(hdf5_path, test_groups, rng), log_dir)
    eval_env = DummyVecEnv([make_env for _ in range(num_envs)])

    try:
        check_env(JassEnv(hdf5_path, test_groups, rng))
        print("Environment check passed.")
    except Exception as e:
        print(f"Env check failed: {e}")

    policy_kwargs = dict(
        features_extractor_class=JassFeatureExtractor,
        features_extractor_kwargs={'embedding_dim': 16},
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )
    policy = ActorCriticPolicy(
        observation_space=eval_env.observation_space,
        action_space=eval_env.action_space,
        lr_schedule=lambda _: 3e-4,
        **policy_kwargs
    )

    bc_trainer = BC(

        observation_space=eval_env.observation_space,
        action_space=eval_env.action_space,
        rng=rng,
        policy=policy,
        demonstrations=train_trans,
        device='auto',
        custom_logger=logger,
    )

    print("Training BC policy...")
    bc_trainer.train(n_batches=10000)

    print("Evaluating policy...")
    returns = rollout.generate_trajectories(bc_trainer.policy, eval_env, rollout.make_sample_until(min_episodes=10), rng)
    mean_ret = np.mean([ep.rews.sum() for ep in returns])
    print(f"Mean return over 20 episodes: {mean_ret:.3f}")

    policy_path = os.path.join(log_dir, "bc_policy.zip")
    bc_trainer.policy.save(policy_path)
    print(f"Policy saved to {policy_path}")

    eval_env.close()
    print("Completed.")

if __name__ == "__main__":
    main()
