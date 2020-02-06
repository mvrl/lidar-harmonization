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

transforms = Compose([LoadNP(), CloudNormalize(), ToTensor()])
config = Config(25, 25, .001)

parser = argparse.ArgumentParser(description="run script for Intensity Correction")
subparsers = parser.add_subparsers(help='sub-command help')

parser_train = subparsers.add_parser('train', help='train the network')
parser_train.set_defaults(func="train",
                          dataset=LidarDataset("dataset/train_dataset.csv",
                                               transform=transforms),
                          config=config,
                          use_valid=True)
parser_train.add_argument('-b', "--baseline", help="run baseline", action="store_true")

parser_eval = subparsers.add_parser('eval', help='evaluate the network')
parser_eval.set_defaults(func="eval",
                         dataset=LidarDataset("dataset/test_dataset.csv",
                                              transform=transforms),
                         transform=transforms)

args = parser.parse_args()


if args.func == "train":
    print("starting training...")
    train(dataset=args.dataset,
          config=args.config,
          use_valid=args.use_valid)

if args.func == "eval":
    print("starting evaluation...")
    evaluate("intensity_dict", args.dataset)
