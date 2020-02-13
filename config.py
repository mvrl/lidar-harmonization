from pathlib import Path
import torch
import torch.nn as nn


class Config:
    def __init__(self, device_num=0):
        self.name = "Intensity Correction"
        self.device = torch.device("cuda:%s" % device_num if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(device_num) if torch.cuda.is_available() else "cpu"

        self.batch_size = 25
        self.num_classes = 1
        self.criterion = nn.SmoothL1Loss() if self.num_classes == 1 else CrossEntropyLoss()
        self.num_workers = 10

