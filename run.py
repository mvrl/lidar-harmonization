import argparse
import numpy as np
import torch
from pathlib import Path

from lidar_dataset import LidarDataset
from torchvision.transforms import Compose

from train import train
from evaluate import evaluate
from config import Config
from util.transforms import LoadNP, CloudNormalize, CloudAugment, CloudJitter, ToTensor



parser = argparse.ArgumentParser(description="run script for Intensity Correction")
group = parser.add_mutually_exclusive_group()
group.add_argument("-t", "--train", help="train network", action="store_true")
group.add_argument("-e", "--eval", help="evaluate network", action="store_true")

args = parser.parse_args()

# input transform =

config = Config(25, 25, .001)

if args.train:
    transforms = Compose([LoadNP(), CloudNormalize(), ToTensor()])
    dataset = LidarDataset("dataset/train_dataset.csv", transform=transforms)
    train(dataset, config, use_valid=True, baseline=False)

if args.eval:
    transforms = Compose([LoadNP(), CloudNormalize(), ToTensor()])
    dataset = LidarDataset("dataset/test_dataset.csv", transform=transforms)
    evaluate("intensity_dict", dataset)
