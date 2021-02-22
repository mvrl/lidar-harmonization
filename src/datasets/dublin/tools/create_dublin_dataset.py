from src.datasets.tools.create_dataset import create_dataset
from src.datasets.dublin.config import config as dublin_config
import time
from torch import tensor, save
from math import ceil

# Example creation script

if __name__=="__main__":
    print(f"Generating dublin dataset at: {dublin_config['save_path']}")
    start = time.time()
    create_dataset(dublin_config)
    print(f"Generation complete, computing weights")
    dataset_csv_path = dataset_config['save_path']
    dataset = LidarDataset(
                Path(dataset_csv_path) / ('train.csv'), 
                ss=dataset_config['use_ss'])
    igroups = ceil(dublin_config['max_intensity']/dublin_config['igroup_size'])
    weights = make_weights_for_balanced_classes(dataset, igroups)  # more modular parameter here
    end = time.time()
    duration = end - start
    print(f"Weights generated in {duration} seconds")
    weights = save(weights, dataset_config['class_weights'])
    print("Weights saved")
    
