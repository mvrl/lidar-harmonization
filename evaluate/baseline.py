import code
import torch
import numpy as np
from dataset.lidar_dataset import LidarDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from util.metrics import create_kde

def second_closest_point_baseline(config=None,
                    dataset_csv=None,
                    transforms=None):

    print(f"Measuring baseline accuracy")
    print("--Second-closest point method")

    suffix = dataset_csv.split("/")[1]
    dataset = LidarDataset(dataset_csv, transform=transforms)

    dataloader = DataLoader(
        dataset,
        batch_size=100,
        num_workers=config.num_workers,
        drop_last=True)

    mae_output = []
    gt_values = []
    fixed_values = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            gt, alt, fid, _ = batch

            # Get intensity of the gt point
            i_gt = gt[:,0,3]
            gt_values.append(i_gt)
            # Output is second closest point
            output = alt[:, 1, 3]
            fixed_values.append(output)
            mae_output.append(torch.mean(torch.abs(i_gt - output)))

    # create kde plots
    gt_values = np.stack([i.numpy() for i in gt_values]).flatten()
    fixed_values = np.stack([i.numpy() for i in fixed_values]).flatten()

    print(f"GT Values: {len(gt_values)}")
    print(f"Fixed Values: {len(fixed_values)}")

    # total measurement
    total_mae_output = np.mean(np.array(mae_output))

    create_kde(gt_values,
               fixed_values,
               "Ground Truth",
               "Fixed Values",
               f"results/{suffix}/kde_baseline.png",
               text=f"MAE: {total_mae_output:.5f}")

    print(f"MAE: {total_mae_output}")
               

                               
                          
                              
        

    
