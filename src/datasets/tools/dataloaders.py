import code
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import Compose

from src.datasets.tools.lidar_dataset import LidarDatasetNP, LidarDatasetHDF5
from src.datasets.tools.transforms import CloudAngleNormalize
from src.datasets.tools.transforms import Corruption, GlobalShift, CloudJitter

import h5py

def get_transforms(config):
    transforms = [CloudAngleNormalize()]

    if config['dataset']['name'] is "dublin":
        # if config['dataset']['shift']:
        #     transforms += [GlobalShift(**config['dataset'])]
        
        transforms += [Corruption(**config['dataset'])] #, CloudJitter()]

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
                        config['dataset']['dataset_path'],
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
""" 
class DistributedWeightedSampler(Sampler):
# https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143/8
    def __init__(self, dataset, num_replicas=None, rank=None, replacement=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.replacement = replacement


    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # get targets (you can alternatively pass them in __init__, if this op is expensive)
        targets = self.dataset.targets
        targets = targets[self.rank:self.total_size:self.num_replicas]
        assert len(targets) == self.num_samples
        weights = self.calculate_weights(targets)

        return iter(torch.multinomial(weights, self.num_samples, self.replacement).tollist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
"""
