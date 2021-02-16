import code
from pathlib import Path
import torch
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.transforms import Compose


from src.datasets.tools.lidar_dataset import LidarDataset, LidarDatasetNP
from src.datasets.tools.transforms import LoadNP, CloudIntensityNormalize, CloudAngleNormalize
from src.datasets.tools.transforms import Corruption, GlobalShift


def make_weights_for_balanced_classes(dataset, nclasses):
    # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    count = [0] * nclasses

    for i in range(len(dataset)):
        ex, label = dataset[i]
        count[label] += 1

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])

    weight = [0] * len(dataset)
    for idx, val in enumerate(dataset):
        weight[idx] = weight_per_class[val[1]]

    # note that weights do not have to sum to 1
    return torch.tensor(weight)

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

    weights = torch.tensor(np.load(dataset_config['class_weights']))
    sampler = WeightedRandomSampler(weights, len(weights))
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

if __name__ == "__main__":
    # generate class weights
    from src.datasets.dublin.config import config as dataset_config
    import time

    start = time.time()
    dataset_csv_path = dataset_config['save_path']
    dataset = LidarDataset(
                Path(dataset_csv_path) / ('train.csv'), 
                ss=dataset_config['use_ss'])
    weights = make_weights_for_balanced_classes(dataset, 103)  # more modular parameter here
    print(len(weights))
    end = time.time()
    duration = end - start
    print(f"Weights generated in {duration} seconds")
    weights = torch.save(weights, dataset_config['class_weights'])

