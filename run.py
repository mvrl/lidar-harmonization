import argparse
import numpy as np
import torch
from pathlib import Path


from torchvision.transforms import Compose

from train import train
from evaluate.generate_map import generate_map
from config import Config
from util.transforms import LoadNP, CloudNormalize, CloudAugment, CloudJitter, ToTensor


parser = argparse.ArgumentParser(description="run script for Intensity Correction")
subparsers = parser.add_subparsers(help='sub-command help')

parser_train = subparsers.add_parser('train', help='train the network')
parser_train.add_argument('-e', '--epochs', help='number of epochs to train')
parser_train.set_defaults(func=train,
                          config=Config(),
                          dataset_csv="dataset/50_10000/train_dataset.csv",
                          transform=Compose([LoadNP(),
                                             CloudNormalize(),
                                             ToTensor()]),
                          epochs=50,                       
                          use_valid=True)

parser_eval = subparsers.add_parser('eval', help='evaluate the network')
parser_eval.add_argument('state_dict', help='state dictionary for the model')
parser_eval.set_defaults(func=None,
                         config=Config(),
                         dataset_csv="dataset/50_10000/test_dataset.csv",
                         transform=Compose([LoadNP(),
                                             CloudNormalize(),
                                             ToTensor()]))

eval_subparsers = parser_eval.add_subparsers(help='evaluation experiment')
map_parser = eval_subparsers.add_parser('gen_map', help='generate map')
map_parser.set_defaults(func=generate_map,
                        dataset=r"dataset/big_tile/")


args = parser.parse_args()

kwargs = vars(args)
func = kwargs['func']
del kwargs['func']

func(**kwargs)
