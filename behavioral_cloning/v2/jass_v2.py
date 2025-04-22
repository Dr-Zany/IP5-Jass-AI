# -*- coding: utf-8 -*-
"""
train_card_policy_bc.py

Trains a card prediction policy using Behavioral Cloning (BC)
based on data parsed into an HDF5 file.

This version uses a class to organize the training process
and leverages a custom HDF5 dataset with DataLoader for
memory-efficient training on large datasets.

Requires compatible versions of torch, gymnasium, stable-baselines3, imitation, and h5py.
"""

import h5py
import numpy as np
import gymnasium as gym
# stable_baselines3 is implicitly used by imitation for policy structure
import stable_baselines3 as sb3
# Import core BC algorithm and data types
from imitation.algorithms import bc
from imitation.data import types
# Need Dataset and DataLoader for custom data loading
from torch.utils.data import Dataset
import torch.utils.data as data_utils # Use alias for DataLoader

import logging
import os
import torch as th # Use th alias for torch conventions
import traceback # To print full error tracebacks
from typing import Tuple, Optional, List, Dict, Any
from sklearn.model_selection import train_test_split


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


# --- Configuration Constants ---
HDF5_FILE_PATH = r'../../../../Training_Data/jass.hdf5' # <--- Path to the HDF5 file from your parser
POLICY_SAVE_PATH = 'jass_bc_card_policy_591_organized' # Model save path

# --- Data Keys (Must match your parser's output HDF5 structure) ---
OBS_KEY = 'state_bits'         # N x 929 boolean array
ACTION_BITS_KEY = 'action_card_bits' # N x 13 boolean array (the played card's representation)

# --- Model Dimensions (Based on your parser) ---
INPUT_DIM = 929 # From your parser's STATE_BITS calculation
CARD_ACTION_DIM = 9 # 9 possible card positions in hand (index 0-8)

# --- Constants from Parser (Needed to interpret state_bits) ---
CARD_BITS = 13 # Bits used to represent a single card in the parser
NUM_CARDS_HISTORY = 32 # Number of history card slots in state (32 * 13 bits)
NUM_CARDS_TABLE = 3  # Number of table card slots in state (3 * 13 bits)
NUM_CARDS_HAND = 9   # Number of hand card slots in state (9 * 13 bits)

# Calculate the starting index of the player's hand bits within the state_bits vector
# Order from parser: History, Table, Hand, Shown, Trump
HAND_START_BIT_INDEX = (NUM_CARDS_HISTORY * CARD_BITS) + (NUM_CARDS_TABLE * CARD_BITS)
# 32 * 13 = 416
# 3 * 13  = 39
# HAND_START_BIT_INDEX = 416 + 39 = 455. The hand bits are from index 455 up to 455 + (9*13) = 455 + 117 = 572.


# --- Default Training Hyperparameters ---
# Can be overridden when instantiating the trainer class
DEFAULT_TRAINING_PARAMS = {
    'validation_split_size': 0.15,
    'random_seed': 42,
    'bc_batch_size': 128,
    'bc_learning_rate': 3e-4,
    'bc_n_epochs': 10,
    'device': th.device("cuda"),
    'dataloader_num_workers': 2, # Start with 4 workers for simplicity, increase for speed
    'dataloader_pin_memory': True,
}


# --- 1. Data Location Collection Helper Function ---
# This function remains outside the class as it's a data loading utility
def collect_bc_sample_locations_from_hdf5(filepath: str) -> Tuple[Optional[List[Tuple[str, int]]], int]:
    """
    Collects locations (group_name, sample_index) for valid BC samples
    from all groups in the HDF5 file. Does NOT load actual data into memory.

    Returns:
        Tuple[Optional[list], int]: A list of (group_name, index_in_group) tuples
                                     and the total count of samples, or None and 0 on failure.
    """
    sample_locations = []
    total_samples_count = 0

    if not os.path.exists(filepath):
        logging.error(f"HDF5 file not found at: {filepath}")
        return None, 0

    try:
        with h5py.File(filepath, 'r') as f:
            # Get all group names
            # game_groups = list(f.keys()) # Process all groups
            #game_groups = list(f.keys()) # Process all groups initially
            game_groups = list(f.keys()) # Example: Process only first 2000 games

            logging.info(f"Found {len(f.keys())} game groups total. Processing {len(game_groups)} groups for locations.")

            if not game_groups:
                logging.error("No game groups found in the HDF5 file.")
                return None, 0

            for i, group_name in enumerate(game_groups):
                if (i + 1) % 200 == 0:
                    logging.info(f"Collecting locations from group {i+1}/{len(game_groups)}: {group_name}")
                try:
                    group = f[group_name]
                    # --- Check required keys (without reading data) ---
                    if OBS_KEY not in group:
                        logging.warning(f"Skipping group '{group_name}': Missing dataset '{OBS_KEY}'")
                        continue
                    if ACTION_BITS_KEY not in group:
                        logging.error(f"CRITICAL: Skipping group '{group_name}': Missing dataset '{ACTION_BITS_KEY}'.")
                        continue

                    # Get dataset sizes and shapes without loading data
                    obs_dset_shape = group[OBS_KEY].shape
                    actions_bits_dset_shape = group[ACTION_BITS_KEY].shape

                    # --- Basic Validation (shape only) ---
                    if obs_dset_shape[0] != actions_bits_dset_shape[0]:
                         logging.warning(f"Skipping group '{group_name}': Data length mismatch ({obs_dset_shape[0]} vs {actions_bits_dset_shape[0]}).")
                         continue
                    if obs_dset_shape[0] == 0:
                         logging.info(f"Skipping empty group '{group_name}'.")
                         continue
                    if obs_dset_shape[1] != INPUT_DIM:
                         logging.warning(f"Skipping group '{group_name}': Observation dimension mismatch (Expected {INPUT_DIM}, Got {obs_dset_shape[1]}).")
                         continue
                    if actions_bits_dset_shape[1] != CARD_BITS:
                         logging.warning(f"Skipping group '{group_name}': Action bits dimension mismatch (Expected {CARD_BITS}, Got {actions_bits_dset_shape[1]}).")
                         continue

                    # Collect locations for all samples in this group that passed initial checks.
                    # The action index derivation (card in hand check) will happen in the Dataset.
                    for j in range(obs_dset_shape[0]):
                        sample_locations.append((group_name, j))

                    total_samples_count += obs_dset_shape[0] # Count all samples from valid groups

                except Exception as e:
                    logging.error(f"Error processing group '{group_name}' for locations: {e}")
                    traceback.print_exc()
                    continue # Skip this group on error

            logging.info(f"Finished collecting locations for {len(game_groups)} groups. Total potential samples found: {len(sample_locations)}")

            return sample_locations, total_samples_count

    except Exception as e:
        logging.critical(f"Failed to access HDF5 file '{filepath}' or process groups for locations: {e}")
        traceback.print_exc()
        return None, 0


# --- 2. Custom HDF5 Dataset Class ---
# This class also remains outside the trainer class as it's a generic data interface
class HDF5Dataset(Dataset):
    """Custom Dataset for reading observations and converting action bits
       to indices from HDF5 file on the fly."""

    def __init__(self, hdf5_filepath: str, sample_locations: List[Tuple[str, int]]):
        """
        Args:
            hdf5_filepath (str): Path to the HDF5 file.
            sample_locations (list): List of (group_name, index_in_group) tuples
                                     representing the samples in this dataset split.
        """
        self.hdf5_filepath = hdf5_filepath
        self.sample_locations = sample_locations
        self._file = None # HDF5 file handle, opened on first access

        # Basic validation of locations structure (optional but good)
        if not all(isinstance(loc, tuple) and len(loc) == 2 for loc in sample_locations):
             logging.warning("Dataset: Sample locations list has unexpected format.")
        if not sample_locations:
             logging.warning("Dataset: Initialized with an empty list of sample locations.")


    def __len__(self):
        """Returns the total number of samples in this dataset split."""
        return len(self.sample_locations)

    def _get_file(self):
        """Helper to open the HDF5 file. Designed to be safer with multiprocessing
           by potentially opening a handle per worker process used by DataLoader."""
        # Check if the file is already open in this process/thread
        # h5py File objects are not thread-safe. Opening per process is safer.
        # A simple check for self._file being None works well with DataLoader's
        # worker initialization if the dataset object is copied to workers.
        if self._file is None:
             try:
                 # Using swmr=True (Single Writer Multiple Reader) might be necessary
                 # if the HDF5 file could potentially be written to while reading,
                 # but for a static file, 'r' mode is fine. swmr requires HDF5 1.9+
                 # self._file = h5py.File(self.hdf5_filepath, 'r', swmr=True)
                 self._file = h5py.File(self.hdf5_filepath, 'r')
                 # logging.debug(f"Opened HDF5 file {self.hdf5_filepath} in process {os.getpid()}") # Optional: log file opening
             except Exception as e:
                  logging.critical(f"Dataset Error: Failed to open HDF5 file {self.hdf5_filepath} in _get_file for process {os.getpid()}: {e}")
                  # In a real scenario, failing to open the file is critical.
                  # Raising an error will stop the DataLoader worker or main process.
                  raise
        return self._file

    def __del__(self):
        """Ensures the HDF5 file is closed when the dataset object is garbage collected."""
        if self._file is not None:
             try:
                 self._file.close()
                 # logging.debug(f"Closed HDF5 file {self.hdf5_filepath} in __del__ for process {os.getpid()}") # Optional: log file closing
             except Exception as e:
                 logging.error(f"Dataset Error: Error closing HDF5 file {self.hdf5_filepath} in __del__: {e}")
             self._file = None # Clear the reference


    def __getitem__(self, idx):
        """
        Reads a single sample from the HDF5 file and returns observation and action index.
        Performs the action bit to index conversion.

        Args:
            idx (int): Index of the sample to retrieve (from 0 to len - 1).

        Returns:
            Tuple[th.Tensor, th.Tensor]: The observation (float32) and
                                         the action index (int64).

        Raises:
            IndexError: If idx is out of the valid range.
            RuntimeError: If data inconsistencies prevent deriving a valid action index.
            Exception: For other errors during file reading.
        """
        if idx < 0 or idx >= len(self.sample_locations):
            raise IndexError(f"Dataset index ({idx}) out of range (0-{len(self.sample_locations)-1})")

        group_name, sample_index_in_group = self.sample_locations[idx]

        # Use the helper to get the file object (opens if not already open in this process)
        f = self._get_file()

        try:
            # Access the group and datasets
            group = f[group_name]
            # Using [index:index+1] slicing reads a single row and keeps the dimension (shape (1, D))
            obs_data_single = group[OBS_KEY][sample_index_in_group:sample_index_in_group+1] # Shape (1, INPUT_DIM) bool
            actions_bits_data_single = group[ACTION_BITS_KEY][sample_index_in_group:sample_index_in_group+1] # Shape (1, CARD_BITS) bool

            # --- Convert Action Card Bits (1, 13) bool to Action Index (scalar) int64 ---
            # Convert boolean arrays to integers for easier comparison/processing
            state_vec_int_single = obs_data_single.astype(np.int8).squeeze(axis=0) # Shape (INPUT_DIM,) int8
            played_card_bits_int_single = actions_bits_data_single.astype(np.int8).squeeze(axis=0) # Shape (CARD_BITS,) int8

            # Initialize action_index BEFORE the loop and BEFORE the check below
            action_index = -1

            # Extract the player's hand bits from the single state vector
            hand_bits_int_single = state_vec_int_single[HAND_START_BIT_INDEX : HAND_START_BIT_INDEX + NUM_CARDS_HAND * CARD_BITS]

            # Find the index of the played_card_bits within the hand_bits
            for card_idx_in_hand in range(NUM_CARDS_HAND):
                start_idx = card_idx_in_hand * CARD_BITS
                end_idx = start_idx + CARD_BITS
                current_hand_card_bits_int = hand_bits_int_single[start_idx : end_idx]

                # Compare the played card bits with the current hand card bits
                if np.array_equal(played_card_bits_int_single, current_hand_card_bits_int):
                    action_index = card_idx_in_hand # Found the index!
                    break # Found the card in hand, its index is our action

            # --- Error Handling for Data Inconsistency ---
            # This check happens AFTER the loop finishes and action_index has been determined (or remained -1).
            if action_index == -1:
                 played_bits_list = played_card_bits_int_single.tolist()
                 hand_bits_list = hand_bits_int_single.reshape(-1, CARD_BITS).tolist()
                 error_msg = (f"Dataset Error: Sample {idx} ({group_name}, {sample_index_in_group}): "
                              f"Played card bits {played_bits_list} not found in hand bits {hand_bits_list}. "
                              "This sample cannot be used for training.")
                 # Log the error for debugging data issues
                 logging.error(error_msg)
                 # Raise a RuntimeError to stop training for this data problem
                 raise RuntimeError(error_msg)


            # Convert numpy arrays to torch tensors - THIS IS *AFTER* the error check
            # Squeeze removes the leading dimension of 1 from the [index:index+1] slice
            obs_tensor = th.from_numpy(obs_data_single.astype(np.float32)).squeeze(axis=0) # Shape (INPUT_DIM,) float32
            action_index_tensor = th.tensor(action_index, dtype=th.int64) # Scalar tensor (int64)

            # Return a dictionary expected by the DataLoader/imitation trainer
            return {"obs": obs_tensor, "acts": action_index_tensor}

        except Exception as e:
            # Catch any exception occurring during reading/processing this specific sample
            logging.error(f"Dataset Error: Error reading or processing sample {idx} ({group_name}, {sample_index_in_group}) in __getitem__: {e}")
            traceback.print_exc()
            # Re-raise the exception to signal the error to the DataLoader/trainer
            raise


# --- 3. Trainer Class ---
# This class encapsulates the training setup and execution
class BCCardPolicyTrainer:
    """
    Encapsulates the Behavioral Cloning training process for the Jass card policy.
    Handles data loading setup, trainer initialization, training, and saving.
    """
    def __init__(self, hdf5_filepath: str, save_path: str, params: Dict[str, Any]):
        """
        Initializes the trainer.

        Args:
            hdf5_filepath (str): Path to the HDF5 file containing training data.
            save_path (str): Base path for saving the trained policy.
            params (Dict[str, Any]): Dictionary of training parameters.
        """
        self.hdf5_filepath = hdf5_filepath
        self.save_path = save_path
        self.params = params # Store parameters

        # Apply default parameters if not provided
        for key, default_value in DEFAULT_TRAINING_PARAMS.items():
            if key not in self.params:
                self.params[key] = default_value

        # Extract parameters for clarity
        self.validation_split_size = self.params['validation_split_size']
        self.random_seed = self.params['random_seed']
        self.bc_batch_size = self.params['bc_batch_size']
        self.bc_learning_rate = self.params['bc_learning_rate']
        self.bc_n_epochs = self.params['bc_n_epochs']
        self.device = self.params['device']
        self.dataloader_num_workers = self.params['dataloader_num_workers']
        self.dataloader_pin_memory = self.params['dataloader_pin_memory']

        logging.info(f"Trainer initialized with parameters: {self.params}")
        logging.info(f"Using device for training: {self.device}")


        self.train_dataloader = None
        # self.val_dataloader = None # Optional: for validation during training
        self.bc_trainer = None
        self.n_batches = 0 # Total training batches calculated later


    def load_and_prepare_data(self):
        """
        Collects sample locations, splits them, and creates HDF5Dataset and DataLoaders.
        """
        logging.info("Loading and preparing data...")

        # --- Collect Data Sample Locations ---
        sample_locations, total_samples = collect_bc_sample_locations_from_hdf5(self.hdf5_filepath)

        # --- Error Handling for Location Collection ---
        if sample_locations is None:
            logging.critical("Trainer Error: Failed to collect sample locations (function returned None). Check HDF5_FILE_PATH and file accessibility. Exiting.")
            return False # Indicate failure

        if not sample_locations:
             logging.critical("Trainer Error: No valid data sample locations were collected (list is empty). Ensure HDF5 file contains groups with required datasets and correct dimensions. Exiting.")
             return False # Indicate failure

        logging.info(f"Total valid data sample locations collected: {len(sample_locations)}.")

        # --- Split Sample Locations ---
        logging.info(f"Splitting sample locations (keeping {1-self.validation_split_size:.0%} for training)...")
        try:
            train_locations, val_locations = train_test_split(
                sample_locations,
                test_size=self.validation_split_size,
                random_state=self.random_seed,
                shuffle=True,
            )
        except ValueError as e:
             logging.critical(f"Trainer Error: Failed to split data locations. This can happen if test_size is too large for the number of samples. Error: {e}. Exiting.")
             return False # Indicate failure

        logging.info(f"Training locations: {len(train_locations)}, Validation locations: {len(val_locations)}")

        # Free up memory from the original full locations list
        del sample_locations

        # --- Create HDF5 Dataset for Training ---
        if not train_locations:
            logging.critical("Trainer Error: No training samples available after splitting locations. Adjust split size or provide more data. Exiting.")
            return False # Indicate failure
        train_dataset = HDF5Dataset(self.hdf5_filepath, train_locations)

        # --- Create DataLoader from the Training Dataset ---
        # This DataLoader will be used by the BC trainer
        self.train_dataloader = data_utils.DataLoader(
            train_dataset,
            batch_size=self.bc_batch_size,
            shuffle=True, # Shuffle data for better training convergence
            num_workers=self.dataloader_num_workers,
            pin_memory=self.dataloader_pin_memory,
            drop_last=True, # <--- Add this line

        )
        logging.info(f"Created training DataLoader with batch_size={self.bc_batch_size}, num_workers={self.dataloader_num_workers}")


        # --- Optional: Create DataLoader for Validation ---
        # You would use this if you configure the BC trainer for validation during training
        # if val_locations:
        #     self.val_dataloader = data_utils.DataLoader(
        #         HDF5Dataset(self.hdf5_filepath, val_locations),
        #         batch_size=self.bc_batch_size, # Validation batch size can be different
        #         shuffle=False, # No need to shuffle validation data
        #         num_workers=self.dataloader_num_workers,
        #         pin_memory=self.dataloader_pin_memory,
        #     )
        #     logging.info(f"Created validation DataLoader with batch_size={self.bc_batch_size}, num_workers={self.dataloader_num_workers}")
        # else:
        #     logging.warning("No validation samples available after splitting locations. Skipping validation DataLoader creation.")
        #     self.val_dataloader = None


        # Calculate total number of training batches needed
        num_train_samples = len(train_dataset)
        n_batches_per_epoch = num_train_samples // self.bc_batch_size
        if num_train_samples % self.bc_batch_size != 0:
            n_batches_per_epoch += 1
        self.n_batches = n_batches_per_epoch * self.bc_n_epochs
        # Ensure n_batches is at least 1 if there are samples
        self.n_batches = max(1, self.n_batches) if num_train_samples > 0 else 0

        logging.info(f"Data preparation complete. Total training batches calculated: {self.n_batches}")
        return True # Indicate success


    def setup_trainer(self):
        """Initializes the imitation.algorithms.bc.BC trainer."""
        if self.train_dataloader is None:
            logging.critical("Trainer Error: Data not loaded. Call load_and_prepare_data() first. Exiting.")
            return False # Indicate failure

        logging.info("Setting up BC trainer...")

        # --- Prepare Spaces for Imitation Library ---
        # These match the expected output of HDF5Dataset.__getitem__ and DataLoader collation
        observation_space = gym.spaces.Box(low=0, high=1, shape=(INPUT_DIM,), dtype=np.float32)
        action_space = gym.spaces.Discrete(CARD_ACTION_DIM) # Discrete(9) for action index

        # --- Initialize Behavioral Cloning Model ---
        rng = np.random.default_rng(self.random_seed)

        try:
            self.bc_trainer = bc.BC(
                observation_space=observation_space,
                action_space=action_space,
                demonstrations=self.train_dataloader, # Pass the DataLoader instance
                batch_size=self.bc_batch_size, # Passed for internal logic/logging? DataLoader controls actual batch size
                optimizer_kwargs=dict(lr=self.bc_learning_rate),
                device=self.device, # Trainer and policy will be moved to this device
                rng=rng,
                # Note: loss_calculator is handled internally by bc.BC using Cross-Entropy Loss for Discrete actions.
                # eval_dataloader=self.val_dataloader, # Pass validation loader if created
                # eval_metrics=..., # Define appropriate metrics if evaluating
            )
            logging.info("BC trainer initialized successfully.")
            return True # Indicate success

        except Exception as e:
            logging.critical(f"Trainer Error: Failed to initialize BC trainer: {e}")
            traceback.print_exc()
            return False # Indicate failure


    def train(self):
        """Runs the Behavioral Cloning training process."""
        if self.bc_trainer is None:
            logging.critical("Trainer Error: Trainer not set up. Call setup_trainer() first. Exiting.")
            return False # Indicate failure

        if self.n_batches == 0:
            logging.warning("Trainer: No training batches calculated. Skipping training.")
            return True # Consider success if no training needed due to lack of data

        logging.info(f"Starting BC training for {self.bc_n_epochs} epochs totaling {self.n_batches} batches...")

        try:
            # bc.BC.train takes total batches to run for.
            # It iterates through the provided DataLoader to get batches.
            log_interval = max(1, self.n_batches // 50) # Log ~50 times during training
            logging.info(f"Logging training progress every {log_interval} batches.")
            self.bc_trainer.train(n_batches=self.n_batches, log_interval=log_interval)

            logging.info("BC training finished.")
            return True # Indicate success

        except Exception as e:
            logging.critical(f"Trainer Error: An error occurred during training: {e}")
            traceback.print_exc()
            # The device mismatch error would likely happen here
            return False # Indicate failure


    def save_policy(self):
        """Saves the trained policy model."""
        if self.bc_trainer is None or self.bc_trainer.policy is None:
            logging.warning("Trainer: No policy to save. Training may have failed or was skipped.")
            return False # Indicate failure

        if self.n_batches == 0: # Only attempt saving if training was intended
             logging.warning("Trainer: Skipping policy save because no training batches were available.")
             return True # Consider success if saving was skipped intentionally

        logging.info(f"Attempting to save policy to {self.save_path}.zip")
        try:
            # The policy object is stored in bc_trainer.policy.
            # Save it using the underlying SB3 policy save method (.zip file).
            save_path_full = f"{self.save_path}.zip"
            self.bc_trainer.policy.save(save_path_full)
            logging.info(f"Policy saved successfully to {save_path_full}")
            return True # Indicate success

        except Exception as e:
            logging.error(f"Trainer Error: Error saving BC policy: {e}")
            traceback.print_exc()
            return False # Indicate failure


    # Optional: Add a load_policy method here if needed for inference later
    # def load_policy(self, model_path: str):
    #     """Loads a trained policy model."""
    #     try:
    #         # Assuming it's an SB3 policy saved with .save()
    #         self.policy = sb3.common.policies.deserialize_policy(model_path, device=self.device)
    #         logging.info(f"Policy loaded successfully from {model_path}")
    #         # You might need observation_space and action_space defined if loading policy standalone
    #     except Exception as e:
    #         logging.error(f"Error loading policy from {model_path}: {e}")
    #         self.policy = None


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Script Start ---")

    # --- Configure and Instantiate Trainer ---
    # Use default parameters, or override specific ones here
    trainer_params = DEFAULT_TRAINING_PARAMS.copy()
    # Example override:
    # trainer_params['bc_n_epochs'] = 20
    # trainer_params['dataloader_num_workers'] = 4 # Increase workers for speed
    # trainer_params['device'] = th.device("cuda:1") # Specify a different GPU if available

    logging.info(f"Using device: {trainer_params['device']}")


    trainer = BCCardPolicyTrainer(
        hdf5_filepath=HDF5_FILE_PATH,
        save_path=POLICY_SAVE_PATH,
        params=trainer_params # Pass configuration parameters
    )

    # --- Execute Training Workflow ---
    # 1. Load and prepare data
    if not trainer.load_and_prepare_data():
        logging.critical("Data preparation failed. Exiting.")
        exit() # Exit if data loading/preparation failed

    # 2. Setup the BC trainer model
    if not trainer.setup_trainer():
         logging.critical("Trainer setup failed. Exiting.")
         exit() # Exit if trainer initialization failed

    # 3. Run the training process
    # The device mismatch error would likely happen inside trainer.train()
    if not trainer.train():
         logging.critical("Training process failed. Exiting.")
         exit() # Exit if training encountered a critical error

    # 4. Save the trained policy
    if not trainer.save_policy():
         logging.error("Policy saving failed.")
         # Do not necessarily exit, training might have succeeded but saving failed


    logging.info("--- Script Finished ---")