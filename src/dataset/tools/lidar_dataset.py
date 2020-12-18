import code
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from torchvision.transforms import Compose


class LidarDataset(Dataset):
    def __init__(self, csv_filepath, transform=None, dual_flight=None, limit=None):
        """ LidarDataset class for Harmonization
            Args:
                csv_filepath (str): path to training set csv
                transform (callable): transforms to be applied to examples
                dual_mode (int): flight_id to be filtered for dual_flight test cases
                                 One of 7, 30, 21, 10, 37, 20,  2, 35,  4, 39,  0, 15.
        """
                            
        self.csv_filepath = csv_filepath
        self.transform = transform
        self.df = pd.read_csv(self.csv_filepath)
        
        if dual_flight:
            self.df = self.df.loc[self.df['flight_num'] == dual_flight]

        if limit is not None:
            self.df = self.df[:limit]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        example  = self.df.iloc[idx]["examples"]
        
        if self.transform:
            example = self.transform(example)

        return example

