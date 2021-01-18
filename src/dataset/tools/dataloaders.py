import code
from src.dataset.tools.lidar_dataset import LidarDataset

from src.dataset.tools.transforms import LoadNP, CloudCenter
from src.dataset.tools.transforms import CloudIntensityNormalize
from src.dataset.tools.transforms import CloudAngleNormalize, GetTargets

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

transforms = Compose([
    LoadNP(), 
    CloudIntensityNormalize(512),
    CloudAngleNormalize(), 
    GetTargets()])

transforms_no_load = Compose([
    CloudIntensityNormalize(512),
    CloudAngleNormalize(),
    GetTargets()])

def get_dataloader(dataset_csv, batch_size, num_workers, drop_last=True, limit=None):

    dataset = LidarDataset(dataset_csv, transform=transforms, limit=limit)

    return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last)

