import argparse
import numpy as np
import torch
from pathlib import Path


from torchvision.transforms import Compose

from train import train
from evaluate.generate_map import generate_map
from evaluate.measure_accuracy import measure_accuracy
from evaluate.baseline import second_closest_point_baseline
from config import Config
from util.transforms import *

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

max_intensity_bound = 512

parser = argparse.ArgumentParser(description="run script for Intensity Correction")
subparsers = parser.add_subparsers(help='sub-command help')

parser_train = subparsers.add_parser('train', help='train the network')
parser_train.add_argument('-b', '--batch_size', type=int, help='batch size')
parser_train.add_argument('-e', '--epochs', type=int, help='number of epochs to train')
parser_train.add_argument('-d', '--dataset_csv', help='training dataset csv to use')
parser_train.add_argument('--no-pass-center', type=str2bool, nargs='?',
                        const=True, default=False,help='don\'t pass center point during training')


parser_train.set_defaults(func=train,
                          config=Config(),
                          transforms=Compose(
                              [LoadNP(),
                               CloudCenter(),
                               CloudIntensityClip(max_intensity_bound),
                               CloudIntensityNormalize(max_intensity_bound),
                               CloudAugment(),
                               ToTensor()]
                          ),
                          epochs=50,                       
                          use_valid=True,
                          no_pass_center=False)

parser_eval = subparsers.add_parser('eval', help='evaluate the network')
parser_eval.add_argument('state_dict', help='state dictionary for the model')
parser_eval.add_argument('dataset_csv', help='testing csv to use')


parser_eval.set_defaults(func=None,
                         config=Config(),
                         transforms=Compose([LoadNP(),
                                             CloudCenter(),
                                             CloudIntensityClip(max_intensity_bound),
                                             CloudIntensityNormalize(max_intensity_bound),
                                             ToTensor()]))

eval_subparsers = parser_eval.add_subparsers(help='evaluation experiment')

map_parser = eval_subparsers.add_parser('gen_map', help="create big map")
map_parser.set_defaults(func=generate_map,
                        dataset=r"dataset/big_tile/")

acc_parser = eval_subparsers.add_parser('measure_acc', help='meausre accuracy on tileset')
acc_parser.set_defaults(func=measure_accuracy)

baseline_parser = subparsers.add_parser('baseline', help='measure baseline')
baseline_parser.add_argument('dataset_csv', help='testing csv to use')

baseline_parser.set_defaults(func=None,
                             config=Config(),
                             transforms=Compose([LoadNP(),
                                                 CloudCenter(),
                                                 CloudIntensityClip(max_intensity_bound),
                                                 CloudIntensityNormalize(max_intensity_bound),
                                                 ToTensor()]))

baseline_subparsers = baseline_parser.add_subparsers(help='baseline methods')
second_closest_point_parser = baseline_subparsers.add_parser(
    'second_closest_point',
    help='baseline with second closest point')

second_closest_point_parser.set_defaults(func=second_closest_point_baseline)
args = parser.parse_args()

# print(args)
kwargs = vars(args)
func = kwargs['func']
del kwargs['func']

func(**kwargs)
