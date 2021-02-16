import code
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from src.datasets.tools.lidar_dataset import LidarDataset, LidarDatasetNP
from src.datasets.tools.transforms import LoadNP, CloudIntensityNormalize, CloudAngleNormalize
from src.datasets.tools.transforms import Corruption, GlobalShift


def get_dataloaders(dataset_config, train_config):
    transforms = [LoadNP(), 
                  CloudAngleNormalize()]

    if dataset_config['name'] is "dublin":
        if dataset_config['shift']:
            transforms.append(GlobalShift(**dataset_config))
        
        transforms.extend([
            Corruption(**dataset_config), 
            CloudIntensityNormalize(dataset_config['max_intensity'])])

    if dataset_config['name'] is "kylidar":
        pass

    transforms = Compose(transforms)

    dataset_csv_path = dataset_config['save_path']
    dataloaders = {}

    for phase in dataset_config['phases']:
        dataloaders[phase] = DataLoader(
            LidarDataset(
                        Path(dataset_csv_path) / (phase + '.csv'), 
                        transform=transforms,
                        ss=dataset_config['use_ss']),
                    batch_size=train_config['batch_size'],
                    shuffle=True,
                    num_workers=train_config['num_workers'],
                    drop_last=True)

    return dataloaders

    
def get_dataloader_nl(dataset, batch_size, num_workers, drop_last=False):
    # During evaluation, neighborhoods may already be in memory, meaning 
    #   LoadNP() and pandas indexing are no longer required. Introducing a 
    #   second get_dataloaders function resolves this case. 

    transforms = [LoadNP(), 
                  CloudIntensityNormalize(dataset_config['max_intensity']), 
                  CloudAngleNormalize()]

    dataset = SimpleDataset(dataset, transforms_no_load)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last)
