import code
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from transforms import LoadNP, CloudNormalize, CloudAugment, CloudJitter, ToTensor
from torchvision.transforms import Compose

class LidarDataset(Dataset):
    def __init__(self, csv_filepath, transform=None):
        self.csv_filepath = csv_filepath
        self.transform = transform
        self.df = pd.read_csv(self.csv_filepath)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        gt, alt, flight_num, flight_file_path = self.df.iloc[idx, 1:].values
        
        sample = (gt, alt, flight_num, flight_file_path)

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__=='__main__':
    # Some brief testing
    transforms = Compose([LoadNP(), CloudNormalize(), ToTensor()])
    dataset = LidarDataset("dataset.csv", transform=transforms)
    print(len(dataset))
    gt, alt = dataset[0]
    print("GT:")
    print(gt.shape)
    print(gt[0])
    print("ALT:")
    print(alt.shape)
    print(alt[0])

