from src.dataset.tools.lidar_dataset import LidarDataset

from src.dataset.tools.transforms import LoadNP, CloudCenter
from src.dataset.tools.transforms import CloudIntensityNormalize
from src.dataset.tools.transforms import CloudAngleNormalize, GetTargets

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

def get_dataloader(dataset_csv, batch_size, num_workers, drop_last=True):
    transforms = Compose([
        LoadNP(), 
        CloudIntensityNormalize(512),
        CloudAngleNormalize(), 
        GetTargets()])

    dataset = LidarDataset(dataset_csv, transform=transforms)

    return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=drop_last)


