from pathlib import Path
import torch
import torch.nn as nn


class Config:
    def __init__(self, epochs, batch_size, learning_rate, device_num=0):
        self.name = "Intensity Correction"
        self.device = torch.device("cuda:%s" % device_num if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(device_num) if torch.cuda.is_available() else "cpu"

        self.num_classes = 2
        self.num_workers = 10

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
