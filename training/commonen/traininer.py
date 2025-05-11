import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class TrainingMonitor:
    def __init__(self):
        self.train_batch_losses = []
        self.train_batch_accuracies = []
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def on_train_batch_end(self, logs=None):
        self.train_batch_losses.append(logs['loss'])
        self.train_batch_accuracies.append(logs['accuracy'])

    def on_train_epoch_end(self, logs=None):
        self.train_losses.append(logs['loss'])
        self.train_accuracies.append(logs['accuracy'])

    def on_val_epoch_end(self, logs=None):
        self.val_losses.append(logs['loss'])
        self.val_accuracies.append(logs['accuracy'])

    def plot(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(2, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(self.train_batch_losses, label='Train Batch Loss')
        plt.title('Batch Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(self.train_batch_accuracies, label='Train Batch Accuracy')
        plt.title('Batch Accuracy')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()

        plt.show()


class Trainer:
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn, accuracy_fn, train_loader: DataLoader, val_loader: DataLoader, model_path, device='cpu'):
        self.model = model
        self.model_path = model_path
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.monitor = TrainingMonitor()

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss, train_accuracy = self._train_epoch(epoch)
            val_loss, val_accuracy = self._validate_epoch(epoch)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            if val_loss <= min(self.monitor.val_losses, default=float('inf')):
                torch.save(self.model.state_dict(), self.model_path)
                print(f"Model saved at epoch {epoch+1} with validation loss {val_loss:.4f}")
            
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_acc = 0

        for state, action in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}"):
            state, action = state.to(self.device), action.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(state)
            action = action.view(-1)
            loss = self.loss_fn(output, action)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_acc = self.accuracy_fn(output, action)
            total_acc += batch_acc.item()
            self.monitor.on_train_batch_end(logs={'loss': loss.item(), 'accuracy': batch_acc.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_acc / len(self.train_loader)
        self.monitor.on_train_epoch_end(logs={'loss': avg_loss, 'accuracy': avg_accuracy})
        return avg_loss, avg_accuracy
    
    def _validate_epoch(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0

        with torch.no_grad():
            for state, action in tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}"):
                state, action = state.to(self.device), action.to(self.device)

                output = self.model(state)
                action = action.view(-1)
                loss = self.loss_fn(output, action)

                total_loss += loss.item()
                batch_acc = self.accuracy_fn(output, action)
                total_acc += batch_acc.item()

        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_acc / len(self.val_loader)
        self.monitor.on_val_epoch_end(logs={'loss': avg_loss, 'accuracy': avg_accuracy})
        return avg_loss, avg_accuracy
