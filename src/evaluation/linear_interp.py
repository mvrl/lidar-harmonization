from scipy.interpolate import RegularGridInterpolator
from torchvision.transforms import Compose
from tqdm import tqdm
from src.dataset.tools.lidar_dataset import LidarDataset
from src.dataset.tools.transforms import LoadNP, CloudIntensityNormalize, CloudAngleNormalize, GetTargets2
# In this method we want to first interpolate over the grid and then use a 
# variety of mapping methods to perform the final harmonization

# I don't see why we can't just use the dataset that's being used for pointnet

def interpolate_intensity(neighborhood):
    interpolator = RegularGridInterpolator(
            neighborhood[2:, :3], 
            neighborhood[:, 3], 
            method='linear')

    return interpolator(neighborhood[1, :3])

# Compare interpolation results with true value at neighborhood[1, 3]

transforms = Compose([
    LoadNP(), 
    CloudIntensityNormalize(512), 
    CloudAngleNormalize(), 
    GetTargets2()])

csvs = {"train": "dataset/150/train.csv",
        "val": "dataset/150/val.csv",
        "test": "dataset/150/test.csv"}

phases = [k for k in csvs]
datasets = {k : LidarDataset(v, transform=transforms) for k, v in csvs.items()}

for example in datasets['val']:
    data, h_target, i_target = example
    print(data.shape)
    print(data[0, 3])
    print(i_target)
    print(h_target)
    print("Cameras:")
    print(data[0, -1])
    print(data[1, -1])

    break
