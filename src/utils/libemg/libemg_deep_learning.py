import os
from libemg.datasets import *
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class DL_input_data(Dataset):
    def __init__(self, windows, classes):
        self.data = torch.tensor(windows, dtype=torch.float32)
        self.classes = torch.tensor(classes, dtype=torch.long)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx]
        label = self.classes[idx]
        return data, label

    def __len__(self):
        return self.data.shape[0]

def make_data_loader(windows, classes, batch_size=64, generator=None):
    # first we make the object that holds the data
    obj = DL_input_data(windows, classes)
    # and now we make a dataloader with that object
    dl = DataLoader(obj,
    batch_size=batch_size,
    shuffle=True,
    collate_fn = collate_fn,
    generator=generator,
    num_workers=0)
    return dl

def collate_fn(batch):
    # this function is used internally by the dataloader (see line 46)
    # it describes how we stitch together the examples into a batch
    signals, labels = [], []
    for signal, label in batch:
        # concat signals onto list signals
        signals += [signal]
        labels += [label]
    # convert back to tensors
    signals = torch.stack(signals)
    labels = torch.stack(labels).long()
    return signals, labels