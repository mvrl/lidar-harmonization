import code
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision.transforms import Compose


class LidarDataset(Dataset):
    def __init__(self, csv_filepath, transform=None, ss=True, limit=None):
        """ LidarDataset class for Harmonization
            Args:
                csv_filepath (str): path to training set csv
                transform (callable): transforms to be applied to examples
                ss (bool): use source-source examples
        """
                            
        self.csv_filepath = csv_filepath
        self.transform = transform
        self.df = pd.read_csv(self.csv_filepath)

        if limit is not None:
            self.df = self.df[:limit]

        if not ss:
            self.df = self.df[self.df.source_scan != self.df.target_scan]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        example  = self.df.iloc[idx]["examples"]
        
        if self.transform:
            example = self.transform(example)

        return example


class LidarDatasetNP(Dataset):
    def __init__(self, data, transform=None, ss=True):
        self.data = data
        self.transform = transform

        if not ss:
            self.df = self.data[self.data[:, 8] != self.data[:, 8]]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.data[idx])
        else:
            return self.data[idx]
