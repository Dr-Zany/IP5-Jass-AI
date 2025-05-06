import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import random
import numpy as np
from tqdm import tqdm
import os
import csv # Import the csv module

# Define a seed for reproducibility
MANUAL_SEED = 42

def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # Set seed for all GPUs
    torch.backends.cudnn.deterministic = True # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False # Disable cudnn benchmark for reproducibility

class JassDataset(Dataset):
    """
    Custom Dataset for loading Jass game states and actions from an HDF5 file.
    """
    def __init__(self, h5_path):
        """
        Initializes the dataset by opening the HDF5 file and indexing samples.

        Args:
            h5_path (str): Path to the HDF5 file containing game data.
        """
        self.file = h5py.File(h5_path, 'r')
        self.groups = list(self.file.keys())
        self.index = []
        # Create a flat index of (group, sample_index_within_group) tuples
        for g in self.groups:
            n = self.file[g]['state'].shape[0]
            for i in range(n):
                self.index.append((g, i))

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.index)

    def __getitem__(self, idx):
        """
        Retrieves a sample (state and action) by its index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the state tensor and action tensor.
        """
        grp, i = self.index[idx]
        state = self.file[grp]['state'][i]
        action = self.file[grp]['action'][i]
        state_tensor = torch.LongTensor(state)

        # Clamp card indices to be within the valid range [0, 70]
        # The last element is the trump index, which is handled separately.
        # Assuming card indices are the first 71 elements.
        if state_tensor.size(0) > 71: # Ensure there are card indices to clamp
             state_tensor[:71].clamp_(0, 70) # Clamp the first 71 elements (card indices)
        elif state_tensor.size(0) == 71: # If only card indices are present
             state_tensor.clamp_(0, 70)


        return state_tensor, torch.tensor(action, dtype=torch.long)

    def close(self):
        """
        Closes the HDF5 file. Should be called when done with the dataset.
        """
        self.file.close()


class JassDNN(nn.Module):
    """
    Deep Neural Network for Jass policy prediction using one-hot encoded inputs.
    """
    def __init__(self, num_cards=71, trump_dim=7, hidden_sizes=[512, 256, 128]):
        """
        Initializes the JassDNN model.

        Args:
            num_cards (int): The number of possible card values (0-70).
            trump_dim (int): The dimension of the one-hot encoded trump suit (7 suits).
            hidden_sizes (list): A list of integers specifying the number of neurons
                                 in each hidden layer.
        """
        super().__init__()
        self.num_cards = num_cards
        # Calculate the total input dimension: 71 card slots * num_cards + trump_dim
        # Assuming state_idx[:, :71] contains 71 card indices.
        total_input = 71 * num_cards + trump_dim

        layers = []
        in_dim = total_input
        # Build the hidden layers with ReLU activation
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        self.hidden = nn.Sequential(*layers)
        # Policy head outputs logits for the 9 possible actions
        self.policy_head = nn.Linear(in_dim, 9)

    def forward(self, state_idx, trump_onehot, legal_mask=None):
        """
        Performs a forward pass through the network.

        Args:
            state_idx (torch.Tensor): Tensor of shape (batch_size, 71) containing
                                      card indices.
            trump_onehot (torch.Tensor): Tensor of shape (batch_size, 7) containing
                                         one-hot encoded trump suit.
            legal_mask (torch.Tensor, optional): Boolean mask of shape (batch_size, 9)
                                                 indicating legal actions. Defaults to None.

        Returns:
            torch.Tensor: Log probabilities of the actions.
        """
        # One-hot encode the card indices
        # Ensure indices are within bounds before one-hot encoding
        card_indices_clamped = state_idx.clamp(0, self.num_cards - 1)
        one_hot_cards = F.one_hot(card_indices_clamped, num_classes=self.num_cards).float()
        # Flatten the one-hot encoded card representations
        flat_cards = one_hot_cards.view(one_hot_cards.size(0), -1)

        # Concatenate flattened cards and trump one-hot encoding
        x = torch.cat([flat_cards, trump_onehot], dim=1)

        # Pass through hidden layers
        x = self.hidden(x)

        # Get policy logits
        logits = self.policy_head(x)

        # Apply legal mask if provided
        if legal_mask is not None:
            logits = logits.masked_fill(~legal_mask, float('-inf'))

        # Compute log softmax for policy probabilities
        policy = F.log_softmax(logits, dim=1)
        return policy

def train(model, dataloader, optimizer, device):
    """
    Trains the model for one epoch with a progress bar.

    Args:
        model (nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (Optimizer): The optimizer to use.
        device (torch.device): The device (cpu or cuda) to train on.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    # Wrap the dataloader with tqdm for a progress bar
    for state, action in tqdm(dataloader, desc="Training", leave=False):
        state = state.to(device)
        # Assuming state tensor structure is [71 card indices, 1 trump index]
        card_idx = state[:, :71]
        trump_idx = state[:, 71]
        # Ensure trump index is within bounds [0, 6] for one-hot encoding
        trump_idx_clamped = trump_idx.clamp(0, 6)
        trump_onehot = F.one_hot(trump_idx_clamped, num_classes=7).float()

        action = action.to(device)

        optimizer.zero_grad()

        # Forward pass
        log_policy = model(card_idx, trump_onehot)

        # Ensure action tensor is 1D for NLL loss
        if action.dim() != 1:
            action = action.view(-1)

        # Calculate loss
        loss = F.nll_loss(log_policy, action)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def test(model, dataloader, device):
    """
    Evaluates the model on the test set with a progress bar.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader for the test data.
        device (torch.device): The device (cpu or cuda) to evaluate on.

    Returns:
        float: The average test loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad(): # Disable gradient calculation for evaluation
        # Wrap the dataloader with tqdm for a progress bar
        for state, action in tqdm(dataloader, desc="Testing", leave=False):
            state = state.to(device)
            # Assuming state tensor structure is [71 card indices, 1 trump index]
            card_idx = state[:, :71]
            trump_idx = state[:, 71]
            # Ensure trump index is within bounds [0, 6] for one-hot encoding
            trump_idx_clamped = trump_idx.clamp(0, 6)
            trump_onehot = F.one_hot(trump_idx_clamped, num_classes=7).float()

            action = action.to(device)

            # Forward pass
            log_policy = model(card_idx, trump_onehot)

            # Ensure action tensor is 1D for NLL loss
            if action.dim() != 1:
                action = action.view(-1)

            # Calculate loss
            loss = F.nll_loss(log_policy, action)
            total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Jass DNN (Policy Only, One-Hot)')
    parser.add_argument('--data', type=str, required=True, help='Path to jass_dataset.hdf5')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training and evaluation.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='Total number of samples to use from the dataset. Use -1 for all samples.')
    parser.add_argument('--test_split', type=float, default=0.1,
                        help='Fraction of data to use for the test set (e.g., 0.1 for 10%).')
    parser.add_argument('--seed', type=int, default=MANUAL_SEED,
                        help=f'Random seed for reproducibility (default: {MANUAL_SEED}).')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save model checkpoints.')
    parser.add_argument('--metrics_csv', type=str, default='training_metrics.csv',
                        help='Path to save the training metrics CSV file.') # New argument
    args = parser.parse_args()

    # Set the random seed for reproducibility
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, 'best_jass_dnn_policy.pth')

    # Load the full dataset
    full_dataset = JassDataset(args.data)

    # Determine the total number of samples to use
    if args.num_samples > 0 and args.num_samples <= len(full_dataset):
        total_samples_to_use = args.num_samples
        # Create a subset of the full dataset if num_samples is specified
        indices = list(range(len(full_dataset)))
        random.shuffle(indices) # Shuffle indices before taking a subset
        subset_indices = indices[:total_samples_to_use]
        dataset_to_split = Subset(full_dataset, subset_indices)
        print(f'Using a subset of {total_samples_to_use} samples from the dataset.')
    else:
        total_samples_to_use = len(full_dataset)
        dataset_to_split = full_dataset
        print(f'Using all {total_samples_to_use} samples from the dataset.')


    # Split the dataset into training and test sets
    test_size = int(total_samples_to_use * args.test_split)
    train_size = total_samples_to_use - test_size
    # random_split uses the PyTorch random seed, which is set by set_seed()
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset_to_split, [train_size, test_size]
    )

    print(f'Splitting data into {len(train_dataset)} training samples and {len(test_dataset)} test samples.')

    # Create DataLoaders for training and test sets
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0) # No need to shuffle test data

    # Initialize the model, optimizer, and move model to device
    model = JassDNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Variable to track the best test loss and save the corresponding model
    best_test_loss = float('inf')

    print(f'Starting training for {args.epochs} epochs...')

    # Open the CSV file for writing metrics
    with open(args.metrics_csv, 'w', newline='') as csvfile:
        metric_writer = csv.writer(csvfile)
        # Write the header row
        metric_writer.writerow(['Epoch', 'Train Loss', 'Test Loss'])

        for epoch in range(1, args.epochs + 1):
            print(f'Epoch {epoch}/{args.epochs}') # Print current epoch number

            # Train for one epoch with progress bar
            train_loss = train(model, train_loader, optimizer, device)

            # Evaluate on the test set with progress bar
            test_loss = test(model, test_loader, device)

            # Print epoch results
            print(f'Epoch {epoch}/{args.epochs}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}')

            # Write metrics to the CSV file
            metric_writer.writerow([epoch, train_loss, test_loss])
            # Flush the buffer to ensure data is written to the file immediately
            csvfile.flush()


            # Check if the current test loss is the best seen so far
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                # Save the model state dictionary
                torch.save(model.state_dict(), best_model_path)
                print(f'Saved best model with Test Loss: {best_test_loss:.4f}')

    print('Training finished.')

    # Close the HDF5 file when done
    full_dataset.close()