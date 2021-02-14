import code
from src.dataset.tools.lidar_dataset import LidarDataset, SimpleDataset

from src.datasets.tools.transforms import LoadNP, CloudCenter
from src.datasets.tools.transforms import CloudIntensityNormalize
from src.datasets.tools.transforms import CloudAngleNormalize, GetTargets

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

def get_dataloader(dataset_csv, batch_size, use_ss, num_workers, drop_last=True, limit=None):

    transforms = Compose([
        LoadNP(),
        CloudIntensityNormalize(512),
        CloudAngleNormalize(),
        GetTargets()])

    dataset = LidarDataset(dataset_csv, transform=transforms, ss=use_ss, limit=limit)
    return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last)


def get_dataloader_nl(dataset, batch_size, num_workers, drop_last=False):

    transforms_no_load = Compose([
        CloudIntensityNormalize(512),
        CloudAngleNormalize(),
        GetTargets()])

    dataset = SimpleDataset(dataset, transforms_no_load)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last)
