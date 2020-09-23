import code
import torch
import numpy as np
from src.dataset.tools.lidar_dataset import LidarDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from src.dataset.tools.metrics import create_kde
from scipy.interpolate import griddata
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path

N_SIZE=20
METHOD='linear'

def do_work(ex_path):
    ex = np.load(ex_path)
    ex[:, :3] -= ex[0, :3]  # center point cloud
    ex[:, 3] /= 512  # normalize intensity 0-1
    try:
        pred = griddata(ex[1:N_SIZE, :3], ex[1:N_SIZE, 3], (0, 0, 0), method=METHOD)
        return (pred, ex[0, 3])
    except:
        pass

def interpolate_baseline(dataset_csv=None):

    df = pd.read_csv(dataset_csv)
    save_path = Path(f"results/{N_SIZE}/")
    save_path.mkdir(parents=True, exist_ok=True)
    predictions = []
    targets = []
    unused_count = 0
    pool = Pool(8)
    for data in tqdm(pool.imap_unordered(do_work, df["examples"]), total=len(df)):
        if data:
            if not np.isnan(data[0]).any():
                predictions.append(data[0])
                targets.append(data[1])
        else:
            unused_count+=1 
    
    predictions = np.array(predictions).squeeze()
    targets = np.array(targets)
    residuals = targets - predictions
    # residuals = residuals[~np.isnan(residuals)]
    MAE = np.mean(np.abs(residuals))
    print(predictions.shape)
    print(targets.shape)
    create_kde(
            targets,
            predictions,
            "Ground Truth",
            "Fixed Values (baseline)",
            save_path / f"baseline_interpolation.png",
            text=f"MAE: {str(MAE)}")

    print(f"MAE: {MAE}")
    print(f"There were {unused_count} unused test examples")

def second_closest_point_baseline(config=None,
                    dataset_csv=None,
                    transforms=None):

    print(f"Measuring baseline accuracy -- closest point")

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
            gt, alt = batch

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
               f"results/kde_baseline_closest_point.png",
               text=f"MAE: {total_mae_output:.5f}")

    print(f"MAE: {total_mae_output}")
               
def average_points_baseline(
        config=None,
        dataset_csv=None,
        transforms=None, 
        neighborhood_size=None):

    print("Measuring baseline accuarcy -- average")
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
            gt, alt = batch

            i_gt = gt[:, 0, 3]
            gt_values.append(i_gt)
            output = torch.mean(alt[: , 1:neighborhood_size+1, 3], dim=1)
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
               f"results/kde_baseline_average_{neighborhood_size}.png",
               text=f"MAE: {total_mae_output:.5f}")

interpolate_baseline("dataset/interpolation_dataset/test_dataset.csv")
