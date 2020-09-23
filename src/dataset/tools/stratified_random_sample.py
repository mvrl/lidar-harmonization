import numpy as np
import torch
from torch.utils.data import Sampler


class StratifiedRandomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return 0  #

