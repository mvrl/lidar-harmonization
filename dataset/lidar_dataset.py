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
        gt, alt, flight_num, flight_path = self.df.iloc[idx, 1:].values
        
        sample = (gt, alt, flight_num, flight_path)

        if self.transform:
            sample = self.transform(sample)

        return sample

class BigMapDataset(Dataset):
    def __init__(self, clouds, flight_ids, transform=None):
        self.clouds = clouds
        self.flight_ids = flight_ids
        self.transform = transform

    def __len__(self):
        return len(self.flight_ids)

    def __getitem__(self, idx):
        sample = (self.clouds[idx], self.flight_ids[idx])
        if self.transform:
            sample = self.transform(sample)

        return sample
        
        

        
