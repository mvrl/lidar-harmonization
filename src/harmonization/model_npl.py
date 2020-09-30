import code
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.harmonization.simple_mlp import SimpleMLP
from src.harmonization.inet_pn1 import IntensityNetPN1


class HarmonizationNet(nn.Module):
    def __init__(self,
                 neighborhood_size=0,  # set N=0 for single flight test cases
                 batch_size=50,
                 num_workers=8,
                 feature_transform=False,
                 results_dir=r"results/"):

        super(HarmonizationNet, self).__init__()

        self.neighborhood_size = neighborhood_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.net = IntensityNetPN1(self.neighborhood_size).float()

    def forward(self, batch):
        return self.net(batch)

