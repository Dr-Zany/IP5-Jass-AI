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

    def on_train_batch_end(self, logs=None, model_name=None):
        if model_name is None:
            model_name = self.model_name
        self.train_batch_losses[model_name].append(logs['loss'])
        self.train_batch_accuracies[model_name].append(logs['accuracy'])

    def on_train_epoch_end(self, logs=None, model_name=None):
        if model_name is None:
            model_name = self.model_name
        self.train_losses[model_name].append(logs['loss'])
        self.train_accuracies[model_name].append(logs['accuracy'])

    def on_val_epoch_end(self, logs=None, model_name=None):
        if model_name is None:
            model_name = self.model_name
        self.val_losses[model_name].append(logs['loss'])
        self.val_accuracies[model_name].append(logs['accuracy'])

    def dump(self, dump_path):
        for model_name in self.train_losses.keys():
            with open(f"{dump_path}/{model_name}_monitor.csv", 'w') as f:
                f.write("train_loss,train_accuracy,val_loss,val_accuracy\n")
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