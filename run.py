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

parser = argparse.ArgumentParser(description="run script for Intensity Correction")
subparsers = parser.add_subparsers(help='sub-command help')

parser_train = subparsers.add_parser('train', help='train the network')
parser_train.add_argument('-e', '--epochs', help='number of epochs to train')
parser_train.set_defaults(func=train,
                          config=Config(),
                          dataset=LidarDataset("dataset/50_10000/train_dataset.csv",
                                               transform=transforms),
                          epochs=50,                       
                          use_valid=True)


parser_eval = subparsers.add_parser('eval', help='evaluate the network')
parser_eval.add_argument('state_dict', help='state dictionary for the model')
parser_eval.set_defaults(func=evaluate, dataset=LidarDataset("dataset/50_10000/test_dataset.csv",
                                              transform=transforms),
                         transform=transforms)

args = parser.parse_args()

kwargs = vars(args)
func = kwargs['func']
del kwargs['func']

func(**kwargs)
