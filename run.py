import argparse
import numpy as np
import torch
from pathlib import Path
import code

from torchvision.transforms import Compose

from train import train
from evaluate.generate_map import generate_map
from evaluate.measure_accuracy import measure_accuracy
from evaluate.baseline import second_closest_point_baseline, average_points_baseline
from config import Config
from util.transforms import *

max_intensity_bound = 512  # move this to global config?

class PathFromDataset:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, p):
        path = Path(p)
        csv_path = "dataset" / path / f"{self.mode}_dataset.csv"
        if csv_path.exists():
            return str(csv_path)
        else:
            exit(f"{csv_path} not found!")

class ModelFromDataset:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, p):
        path = Path(p)
        model_path = "results" / path / f"model{self.mode}.pt"
        if model_path.exists():
            return str(model_path)
        else:
            exit(f"{model_path} not found!")
    
    
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



parser = argparse.ArgumentParser(description="run script for Intensity Correction")
subparsers = parser.add_subparsers(help='sub-command help')

parser_train = subparsers.add_parser('train', help='train the network')
parser_train.add_argument('dataset_csv', help='training dataset csv to use')
parser_train.add_argument('neighborhood_size', help='neighborhood size')
parser_train.add_argument('-b', '--batch_size', type=int, help='batch size')
parser_train.add_argument('-e', '--epochs', type=int, help='number of epochs to train')

parser_train.add_argument(
        '-d', 
        '--dual_flight', 
        type=str2bool, 
        nargs='?', 
        const=True,
        default=False,
        help="Use dual-flight dataset")

parser_train.set_defaults(func=train,
                          get_dataset=PathFromDataset('train'),
                          config=Config(),
                          transforms=Compose(
                              [LoadNP(),
                               CloudCenter(),                               
                               CloudIntensityNormalize(max_intensity_bound),
                               ToTensor()]
                          ),
                          val_transforms=Compose([
                              LoadNP(),
                              CloudCenter(),
                              CloudIntensityNormalize(max_intensity_bound),
                              ToTensor()]),                            
                          epochs=50,                       
                          phases=["train", "val"],
                          dual_flight=False)


parser_eval = subparsers.add_parser('eval', help='evaluate the network')
parser_eval.add_argument('neighborhood_size', type=int, help='neighborhood size')
parser_eval.set_defaults(
        func=None,
        config=Config(),
        transforms=Compose([
            LoadNP(),
            CloudCenter(),
            CloudIntensityNormalize(max_intensity_bound),
            ToTensor()]
        ),
        dual_flight=False
    )

eval_subparsers = parser_eval.add_subparsers(help='evaluation experiments')

map_parser = eval_subparsers.add_parser('gen_map', help="create big map")
map_parser.add_argument('state_dict', help="model to use for map harmonization")
map_parser.set_defaults(func=generate_map,
                        tileset_directory=r"dataset/big_tile/")

acc_parser = eval_subparsers.add_parser('measure_acc', help='meausre accuracy on tileset')
acc_parser.add_argument('dataset_csv', help='testing csv to use')
acc_parser.add_argument(
        '-d','--dual_flight', type=str2bool,
        nargs='?', const=True, default=False, help="use dual-flight dataset")
acc_parser.set_defaults(
        func=measure_accuracy,
        get_model=True,
        get_dataset=PathFromDataset('test'))

baseline_parser = subparsers.add_parser('baseline', help='measure baseline')
baseline_parser.add_argument('dataset_csv', help='testing csv to use')

baseline_parser.set_defaults(
        func=None,
        config=Config(),
        get_dataset=PathFromDataset('test'),
        transforms=Compose([
            LoadNP(),
            CloudCenter(),         
            CloudIntensityNormalize(max_intensity_bound),
            ToTensor()]))

baseline_subparsers = baseline_parser.add_subparsers(help='baseline methods')
second_closest_point_parser = baseline_subparsers.add_parser(
    'second_closest_point',
    help='baseline with second closest point')

second_closest_point_parser.set_defaults(func=second_closest_point_baseline)

average_points_parser = baseline_subparsers.add_parser(
        "average_neighborhood", help="baseline via averaging")
average_points_parser.add_argument("neighborhood_size", type=int, help="neighborhood size")
average_points_parser.set_defaults(func=average_points_baseline)



args = parser.parse_args()

# handle various options before submitting to function
kwargs = vars(args)
print(kwargs)
func = kwargs['func']
del kwargs['func']

if 'dataset_csv' in kwargs:
    kwargs['dataset_csv'] = kwargs['get_dataset'](kwargs['dataset_csv'])
    del kwargs['get_dataset']

if 'get_model' in kwargs:
    neighborhood_size = kwargs['neighborhood_size']
    dataset = kwargs['dataset_csv']
    is_df = kwargs['dual_flight']
    if is_df and neighborhood_size == 0:
        model_path = f"results/current/{neighborhood_size}_sf/model.pt" 
    elif is_df and neighborhood_size > 0:
        model_path = f"results/current/{neighborhood_size}_df/model.pt"
    else:
        model_path = f"results/current/{neighborhood_size}_mf/model.pt"
    kwargs['state_dict'] = model_path
    del kwargs['get_model']

func(**kwargs)
