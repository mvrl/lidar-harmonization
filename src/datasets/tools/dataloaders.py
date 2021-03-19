import code
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import Compose

from src.datasets.tools.lidar_dataset import LidarDataset, LidarDatasetNP, LidarDatasetHDF5
from src.datasets.tools.transforms import LoadNP, CloudIntensityNormalize, CloudAngleNormalize
from src.datasets.tools.transforms import Corruption, GlobalShift

import h5py

def get_transforms(config):
    transforms = [ # LoadNP(), 
                  CloudAngleNormalize()]

    if config['dataset']['name'] is "dublin":
        if config['dataset']['shift']:
            transforms.append(GlobalShift(**config['dataset']))
        
        transforms.extend([
            Corruption(**config['dataset']), 
            CloudIntensityNormalize(config['dataset']['max_intensity'])])

    if config['dataset']['name'] is "kylidar":
        pass

    return Compose(transforms)

def get_dataloaders(config):
    
    transforms = get_transforms(config)
    weights = torch.load(config['dataset']['class_weights'])
    shuffle=False

    # dataset_csv_path = config['dataset']['save_path']
    dataloaders = {}

    for phase in config['dataset']['phases']:
        if phase is "train":
            s = WeightedRandomSampler(weights, len(weights))
            shuffle=False
        else:
            s = None
            shuffle=False
        
        dataloaders[phase] = DataLoader(
            LidarDatasetHDF5(
                        config['dataset']['hdf5_path'],
                        transform=transforms,
                        mode=phase,
                        ss=config['dataset']['use_ss']),
                    batch_size=config['train']['batch_size'],
                    sampler=s,
                    shuffle=shuffle,
                    num_workers=config['train']['num_workers'],
                    drop_last=True)

    # if ('eval_save_path' in config['dataset'] and 
    #         (Path(config['dataset']['eval_save_path']) / 'eval_dataset.csv').exists()):
    #     dataloaders['eval'] = DataLoader(
    #         LidarDatasetHDF5(
    #                     Path(config['dataset']['eval_save_path']) / ('eval_dataset.csv'), 
    #                     transform=transforms,
    #                     ss=config['dataset']['use_ss']),
    #                 batch_size=config['train']['batch_size'],
    #                 sampler=None,
    #                 shuffle=False,
    #                 num_workers=config['train']['num_workers'],
    #                 drop_last=False)

    return dataloaders
 
