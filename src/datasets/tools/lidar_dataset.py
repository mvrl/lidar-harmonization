import code
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from torchvision.transforms import Compose
from src.datasets.dublin.config import config as dublin_config


class LidarDatasetNP(Dataset):
    def __init__(self, data, transform=None, ss=True):
        self.data = data
        self.transform = transform

        if not ss:
             # self.df = self.data[self.data[:, 8] != self.data[:, 8]]
             pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx])
        else:
            return self.data[idx]

class LidarDatasetHDF5(Dataset):
    def __init__(self, path, mode='train',transform=None, ss=True):
        self.path = path
        self.file_obj = h5py.File(path, "a")
        self.dataset = self.file_obj[mode]
        self.transform = transform
        self.ss = ss  # not sure how to implement this here :\

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        
        if self.transform:
            ex = self.transform(ex)

        return ex



