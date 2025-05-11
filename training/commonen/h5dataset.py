from torch.utils.data import Dataset
import torch
import h5py
import numpy as np

numpy_to_torch_dtype_dict = {
    np.uint8      : torch.long,
    np.int8       : torch.long,
    np.int16      : torch.long,
    np.int32      : torch.long,
    np.int64      : torch.long,
    np.float16    : torch.float,
    np.float32    : torch.float,
    np.float64    : torch.float,
}

class H5Dataset(Dataset):
    def __init__(self, path, offset=0, length=None):
        self.path = path
        self.h5file = None # will be opened in __getitem__ because it is not pickleable
        self.offset = offset
        self.length = length
        tmp = h5py.File(path, 'r')
        self.state_type = numpy_to_torch_dtype_dict[tmp['states'].dtype.type]
        self.action_type = numpy_to_torch_dtype_dict[tmp['actions'].dtype.type]
        if length is None:
            self.length = tmp.attrs['total_states_saved']

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.h5file is None:
            self.h5file = h5py.File(self.path, 'r')

        state = self.h5file['states'][idx + self.offset]
        action = self.h5file['actions'][idx + self.offset]

        return torch.tensor(state, dtype=self.state_type), torch.tensor(action, dtype=self.action_type)
    
    def __del__(self):
        if self.h5file is not None:
            self.h5file.close()
            self.h5file = None

    def split(self, split):
        """
        Splits the dataset into train and validation sets.
        :param split: float between 0 and 1
        :return: train and validation datasets
        """
        assert 0 < split < 1, "Split must be between 0 and 1"
        train_length = int(self.length * split)
        val_length = self.length - train_length
        train_dataset = H5Dataset(self.path, self.offset, train_length)
        val_dataset = H5Dataset(self.path, self.offset + train_length, val_length)
        return train_dataset, val_dataset
        

            