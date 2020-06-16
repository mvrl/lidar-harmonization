import code
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision.transforms import Compose


class LidarDataset(Dataset):
    def __init__(self, csv_filepath, transform=None):
        self.csv_filepath = csv_filepath
        self.transform = transform
        self.df = pd.read_csv(self.csv_filepath)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        gt, alt, _ = self.df.iloc[idx, 1:].values

        if self.transform:
            sample = self.transform((gt, alt))

        return sample

class BigMapDataset(Dataset):
    def __init__(self, gt_cloud, alt_cloud, transform=None):
        self.gt_cloud = gt_cloud
        self.alt_cloud = alt_cloud
        self.transform = transform

    def __len__(self):
        return len(self.gt_cloud)

    def __getitem__(self, idx):
        sample = self.gt_cloud[idx], self.alt_cloud[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample
        
        

        
