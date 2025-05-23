import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class TrainingMonitor:
    def __init__(self):
        self.model_name = "default"
        self.train_batch_losses = {}
        self.train_batch_accuracies = {}
        self.train_losses = {}
        self.train_accuracies = {}
        self.val_losses = {}
        self.val_accuracies = {}

    def set_model_name(self, model_name):
        self.model_name = model_name
        self.train_batch_losses[model_name] = []
        self.train_batch_accuracies[model_name] = []
        self.train_losses[model_name] = []
        self.train_accuracies[model_name] = []
        self.val_losses[model_name] = []
        self.val_accuracies[model_name] = []

    def on_train_batch_end(self, logs=None):
        self.train_batch_losses[self.model_name].append(logs['loss'])
        self.train_batch_accuracies[self.model_name].append(logs['accuracy'])

    def on_train_epoch_end(self, logs=None):
        self.train_losses[self.model_name].append(logs['loss'])
        self.train_accuracies[self.model_name].append(logs['accuracy'])

    def on_val_epoch_end(self, logs=None):
        self.val_losses[self.model_name].append(logs['loss'])
        self.val_accuracies[self.model_name].append(logs['accuracy'])

    def dump(self, dump_path):
        for model_name in self.train_losses.keys():
            with open(f"{dump_path}/{model_name}_monitor.csv", 'w') as f:
                f.write("train_batch_loss,train_batch_accuracy,train_loss,train_accuracy,val_loss,val_accuracy\n")
                for i in range(len(self.train_losses[model_name])):
                    f.write(f"{self.train_losses[model_name][i]},{self.train_accuracies[model_name][i]},{self.val_losses[model_name][i]},{self.val_accuracies[model_name][i]}\n")



    def plot(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(2, 2, 1)
        for model_name in self.train_losses.keys():
            plt.plot(self.train_losses[model_name], label=f'Train Loss {model_name}')
            plt.plot(self.val_losses[model_name], label=f'Validation Loss {model_name}')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 2)
        for model_name in self.train_accuracies.keys():
            plt.plot(self.train_accuracies[model_name], label=f'Train Accuracy {model_name}')
            plt.plot(self.val_accuracies[model_name], label=f'Validation Accuracy {model_name}')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 2, 3)
        for model_name in self.train_batch_losses.keys():
            plt.plot(self.train_batch_losses[model_name], label=f'Train Batch Loss {model_name}')
        plt.title('Batch Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 2, 4)
        for model_name in self.train_batch_accuracies.keys():
            plt.plot(self.train_batch_accuracies[model_name], label=f'Train Batch Accuracy {model_name}')
        plt.title('Batch Accuracy')
        plt.xlabel('Batch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()

        plt.show()


class Trainer:
    def __init__(self, loss_fn, accuracy_fn, train_loader: DataLoader, val_loader: DataLoader, model_path, device='cpu'):
        self.model_path = model_path
        self.loss_fn = loss_fn
        self.accuracy_fn = accuracy_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.monitor = TrainingMonitor()

    def train(self, epochs, model, model_name="default", optimizer=None):
        self.monitor.set_model_name(model_name)
        for epoch in range(epochs):
            train_loss, train_accuracy = self._train_epoch(epoch, model, optimizer)
            val_loss, val_accuracy = self._validate_epoch(epoch, model)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            if val_loss < min(self.monitor.val_losses[model_name][:-1], default=float('inf')):
                torch.save(model.state_dict(), self.model_path + f"/{model_name}.pth")
                print(f"Model saved at epoch {epoch+1} with validation loss {val_loss:.4f}")
            else:
                break
            
    def _train_epoch(self, epoch, model, optimizer):
        model.train()
        total_loss = 0.0
        total_acc = 0

        for state, action in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}"):
            state, action = state.to(self.device), action.to(self.device)

            optimizer.zero_grad()
            output = model(state)
            action = action.view(-1)
            loss = self.loss_fn(output, action)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_acc = self.accuracy_fn(output, action)
            total_acc += batch_acc.item()
            self.monitor.on_train_batch_end(logs={'loss': loss.item(), 'accuracy': batch_acc.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        avg_accuracy = total_acc / len(self.train_loader)
        self.monitor.on_train_epoch_end(logs={'loss': avg_loss, 'accuracy': avg_accuracy})
        return avg_loss, avg_accuracy
    
    def _validate_epoch(self, epoch, model):
        model.eval()
        total_loss = 0.0
        total_acc = 0

        with torch.no_grad():
            for state, action in tqdm(self.val_loader, desc=f"Validation Epoch {epoch+1}"):
                state, action = state.to(self.device), action.to(self.device)

                output = model(state)
                action = action.view(-1)
                loss = self.loss_fn(output, action)

                total_loss += loss.item()
                batch_acc = self.accuracy_fn(output, action)
                total_acc += batch_acc.item()

        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = total_acc / len(self.val_loader)
        self.monitor.on_val_epoch_end(logs={'loss': avg_loss, 'accuracy': avg_accuracy})
        return avg_loss, avg_accuracy
