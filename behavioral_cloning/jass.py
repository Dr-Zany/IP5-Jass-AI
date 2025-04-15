# -*- coding: utf-8 -*-
"""
train_card_policy_bc.py

Trains a card prediction policy using Behavioral Cloning (BC)
based on data parsed into an HDF5 file.

Assumes the HDF5 file contains 'state_bits' and 'action_index' datasets.
"""

import h5py
import numpy as np
import gymnasium as gym
from sklearn.model_selection import train_test_split
import stable_baselines3 as sb3
# Use PPO or another SB3 algorithm class compatible with BC policy structure if needed
from stable_baselines3 import PPO
from imitation.algorithms import bc
from imitation.data import types
import logging
import os
import torch
from typing import Tuple, Optional

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

# --- Configuration ---
HDF5_FILE_PATH = r'../../../Training_Data/jass.hdf5'  # <--- Path to the HDF5 file from your parser
POLICY_SAVE_PATH = 'jass_bc_card_policy_591' # Model save path

# --- Data Keys (Must match your parser's output HDF5 structure) ---
OBS_KEY = 'state_bits'
ACTION_INDEX_KEY = 'action_index' # <--- ASSUMED KEY FOR CARD INDEX (0-8)

# --- Model Dimensions (Based on your parser) ---
INPUT_DIM = 591 # From your parser's STATE_BITS calculation
CARD_ACTION_DIM = 9  # 9 possible cards in hand

# --- Training Hyperparameters ---
VALIDATION_SPLIT_SIZE = 0.15 # Use 15% of data for validation (optional for BC eval)
RANDOM_SEED = 42 # For reproducible train/test splits
BC_BATCH_SIZE = 128
BC_LEARNING_RATE = 3e-4
BC_N_EPOCHS = 10 # Number of passes over the training data

device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# --- 1. Load Data Function ---
def load_bc_data_from_hdf5(filepath: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Loads observations and action indices from all groups in the HDF5 file."""
    all_obs = []
    all_actions = []

    if not os.path.exists(filepath):
        logging.error(f"HDF5 file not found at: {filepath}")
        return None, None

    try:
        with h5py.File(filepath, 'r') as f:
            game_groups = list(f.keys())
            logging.info(f"Found {len(game_groups)} game groups in HDF5 file for BC.")

            if not game_groups:
                logging.error("No game groups found in the HDF5 file.")
                return None, None

            for i, group_name in enumerate(game_groups):
                if (i + 1) % 100 == 0:
                    logging.info(f"Loading group {i+1}/{len(game_groups)} for BC: {group_name}")
                try:
                    group = f[group_name]
                    # --- Check required keys ---
                    if OBS_KEY not in group:
                        logging.warning(f"BC Skipping group '{group_name}': Missing dataset '{OBS_KEY}'")
                        continue
                    if ACTION_INDEX_KEY not in group:
                        logging.error(f"BC CRITICAL: Skipping group '{group_name}': Missing dataset '{ACTION_INDEX_KEY}'.")
                        logging.error("Ensure your parser saves the card index (0-8) as 'action_index'.")
                        continue # Skip group if critical action index is missing

                    obs = group[OBS_KEY][:]
                    actions = group[ACTION_INDEX_KEY][:]

                    # --- Basic Validation ---
                    if obs.shape[0] != actions.shape[0]:
                        logging.warning(f"BC Skipping group '{group_name}': Data length mismatch ({obs.shape[0]} vs {actions.shape[0]}).")
                        continue
                    if obs.shape[0] == 0:
                         logging.info(f"BC Skipping empty group '{group_name}'.")
                         continue
                    if obs.shape[1] != INPUT_DIM:
                        logging.warning(f"BC Skipping group '{group_name}': Observation dimension mismatch (Expected {INPUT_DIM}, Got {obs.shape[1]}).")
                        continue
                    if np.max(actions) >= CARD_ACTION_DIM or np.min(actions) < 0:
                         logging.warning(f"BC Skipping group '{group_name}': Action index out of range (0-{CARD_ACTION_DIM-1}). Max: {np.max(actions)}, Min: {np.min(actions)}")
                         continue

                    # Convert boolean observations to float32
                    all_obs.append(obs.astype(np.float32))
                    all_actions.append(actions.astype(np.int64)) # Ensure actions are int64

                except Exception as e:
                    logging.error(f"BC Error processing group '{group_name}': {e}")
                    continue # Skip this group on error

        if not all_obs:
            logging.error("BC: No valid data loaded from any group.")
            return None, None

        # Concatenate data from all groups
        logging.info("BC: Concatenating data from all groups...")
        final_obs = np.concatenate(all_obs, axis=0)
        final_actions = np.concatenate(all_actions, axis=0)
        logging.info("BC: Concatenation complete.")

        return final_obs, final_actions

    except Exception as e:
        logging.error(f"BC: Failed to load data from HDF5 file '{filepath}': {e}")
        return None, None

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Card Policy Training (BC) ---")

    # --- Load Data ---
    observations, card_actions = load_bc_data_from_hdf5(HDF5_FILE_PATH)
    if observations is None:
        logging.critical("Could not load data for BC training. Exiting.")
        exit()
    logging.info(f"Total data loaded for BC: {len(observations)} samples.")

    # --- Split Data (Optional for BC, but good practice) ---
    logging.info(f"Splitting data for BC (keeping {1-VALIDATION_SPLIT_SIZE:.0%} for training)...")
    # We primarily need the training split for BC demonstrations
    obs_train, obs_val, act_train, act_val = train_test_split(
        observations, card_actions,
        test_size=VALIDATION_SPLIT_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    logging.info(f"BC Training samples: {len(obs_train)}, Validation samples: {len(obs_val)}")
    del observations, card_actions, obs_val, act_val # Free up memory

    # --- Prepare Data & Spaces for Imitation Library ---
    train_demonstrations = {"obs": obs_train, "actions": act_train}
    del obs_train, act_train # Free memory

    observation_space = gym.spaces.Box(low=0, high=1, shape=(INPUT_DIM,), dtype=np.float32)
    action_space = gym.spaces.Discrete(CARD_ACTION_DIM)

    # --- Train Behavioral Cloning Model ---
    rng = np.random.default_rng(RANDOM_SEED)
    bc_trainer = bc.BC(
        observation_space=observation_space,
        action_space=action_space,
        demonstrations=train_demonstrations,
        batch_size=BC_BATCH_SIZE,
        optimizer_kwargs=dict(lr=BC_LEARNING_RATE),
        device=device,
        rng=rng
    )

    n_batches = (len(train_demonstrations['obs']) // BC_BATCH_SIZE) * BC_N_EPOCHS
    logging.info(f"Starting BC training for ~{BC_N_EPOCHS} epochs ({n_batches} batches)...")
    bc_trainer.train(n_batches=n_batches, log_interval=max(1, n_batches // 20))

    # --- Save Policy ---
    try:
        bc_trainer.policy.save(POLICY_SAVE_PATH)
        logging.info(f"Card prediction policy saved successfully to {POLICY_SAVE_PATH}.zip")
    except Exception as e:
        logging.error(f"Error saving BC policy: {e}")

    logging.info("--- Card Policy Training Finished ---")
