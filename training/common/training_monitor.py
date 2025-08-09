import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import ceil
import os

class TrainingMonitor:
    def __init__(self):
        self.epoch_data_validation = {}
        self.batch_data = {}
        self.epoch_data = {}

    def _add_structure_if_not_exists(self, data, key, model_name, unit):
        if key not in data:
            data[key] = {}
            data[key]['unit'] = unit
        if model_name not in data[key]:
            data[key][model_name] = []

    def on_train_batch_end(self, model_name, key, value, unit=None):
        self._add_structure_if_not_exists(self.batch_data, key, model_name, unit)
        self.batch_data[key][model_name].append(value)

    def on_train_epoch_end(self, model_name, key, value, unit=None):
        self._add_structure_if_not_exists(self.epoch_data, key, model_name, unit)
        self.epoch_data[key][model_name].append(value)

    def on_val_epoch_end(self, model_name, key, value, unit=None):
        self._add_structure_if_not_exists(self.epoch_data_validation, key, model_name, unit)
        self.epoch_data_validation[key][model_name].append(value)

    def dump(self, dump_path, filename="monitor"):
        def _dump(data, prefix):
            for key, values in data.items():
                with open(f"{dump_path}/{filename}_{prefix}_{key}.csv", "w") as f:
                    for model_name, value_list in values.items():
                        if model_name == "unit":
                            continue
                        f.write(f"{model_name}," + ",".join(map(str, value_list)) + "\n")

        _dump(self.epoch_data, "epoch")
        _dump(self.epoch_data_validation, "val_epoch")
        _dump(self.batch_data, "batch")

    def load(self, dump_path, filename="monitor"):
        def _load(data, prefix):
            # go throug all files in the dump_path with the prefix
            for file_name in os.listdir(dump_path):
                if not file_name.startswith(f"{filename}_{prefix}_") or not file_name.endswith(".csv"):
                    continue
                key = file_name[len(f"{filename}_{prefix}_"):-len(".csv")]
                with open(f"{dump_path}/{file_name}", "r") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        model_name = parts[0]
                        values = list(map(float, parts[1:]))
                        unit = "percent" if key == "accuracy" else None
                        self._add_structure_if_not_exists(data, key, model_name, unit)
                        data[key][model_name] = values

        _load(self.epoch_data, "epoch")
        _load(self.epoch_data_validation, "val_epoch")  
        _load(self.batch_data, "batch")

    def plot(self):
        def _plot_metric(ax, data, title, xlabel, ylabel):
            for model_name, values in data.items():
                if model_name == "unit":
                    continue
                ax.plot(values, label=model_name)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)

        def _plot_key(data, idx, gs, title, xlabel, span_cols=1):
            for key, values in data.items():
                row = idx // 2
                col = idx % 2
                if span_cols == 2:
                    ax = fig.add_subplot(gs[row, :])
                else:
                    ax = fig.add_subplot(gs[row, col])
                ylabel = f"{key} in {values['unit']}" if values['unit'] != None else f"{key}"
                _plot_metric(ax, values, f"{title} - {key}", xlabel, ylabel=ylabel)
                idx += span_cols
            return idx

        n_epoch = len(self.epoch_data) # 2 prints per row
        n_val = len(self.epoch_data_validation) # 2 prints per row
        n_batch = len(self.batch_data) # 1 print per row
        rows = ceil(n_epoch / 2) + ceil(n_val / 2) + n_batch

        fig = plt.figure(figsize=(12, 4 * rows))
        gs = gridspec.GridSpec(rows, 2)

        idx = 0
        _plot_key(self.epoch_data, idx, gs, "Training", "Epoch", span_cols=1)
        idx += n_epoch + n_epoch % 2  # Ensure Validation starts on a new row if n_epoch is odd
        _plot_key(self.epoch_data_validation, idx, gs, "Validation", "Epoch", span_cols=1)
        idx += n_val + n_val % 2 # Ensure Batch starts on a new row if n_val is odd
        _plot_key(self.batch_data, idx, gs, "Training", "Batch", span_cols=2)

        plt.tight_layout()
        plt.show()