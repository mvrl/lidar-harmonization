import os
import json
import time
import code
from pathlib import Path
from datetime import datetime as dt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dataset.lidar_dataset import LidarDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import gaussian_kde

from pptk import kdtree, viewer
from laspy.file import File
from util.apply_rf import apply_rf
from util.metrics import create_kde

from model import IntensityNet

def measure_accuracy(config=None,
                     state_dict=None,
                     tile_directory=None,
                     dataset_csv=None,
                     transforms=None,
                     use_second_point=None,
                     neighborhood_size=None,
                     dual_flight=None,
                     **kwargs):

    start_time = time.time()
    if dual_flight:
        dataset_csv = dataset_csv[:-4] + "_df.csv"
    dataset = LidarDataset(dataset_csv, transform=transforms)
    input_features = 8
    if neighborhood_size == 0:
        save_suffix = "_sf"
    elif neighborhood_size > 0 and dual_flight:
        save_suffix = "_df"
    elif neighborhood_size > 0:
        save_suffix = "_mf"
    else:
        exit("ERROR: this does not appear to be a valid configuration, check neighborhood size: {neighborhood_size} and dual_flight: {dual_flight}")


    print(f"measuring accuracy on {dataset_csv}")
    print(f"using model {state_dict}")

    results_path = Path(f"results/current/{neighborhood_size}{save_suffix}")
    dataloader = DataLoader(
        dataset,
        batch_size=100,
        num_workers=config.num_workers,
        drop_last=True)
    fid_values = []
    mae_output = []
    alt_values = []
    fixed_values = []
    gt_values = []

    # code.interact(local=locals())
    model = IntensityNet(
        num_classes=config.num_classes,
        input_features=input_features).to(config.device).double()
    model.load_state_dict(torch.load(state_dict))
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            eval_time = time.time()          
            
            # Each batch is a (2+150, 9) point cloud, where the first two elements
            # are the ground truth center and an altered copy of the g.t. center.
            # These must be removed prior to training, unless the single-copy 
            # experiment is being run.

            # Get the intensity of the ground truth point
            i_gt = batch[:,0,3].to(config.device)
            i_alt = batch[:,1,3].to(config.device)
            
            if neighborhood_size == 0:
                # the altered flight is just a copy of the gt center
                alt = batch[:, 1, :]
                alt = alt.unsqueeze(1)  # (B, 9) -> (B, 1, 9)
            else:
                # chop out the target point
                alt = batch[:, 2:neighborhood_size+1, :]

            # Extract the pt_src_id
            fid = alt[:, :, 8][:, 0].long().to(config.device)
            alt = alt[:, :, :8]  # remove pt_src_id from input feature vector
            my_fid = int(fid[0])
            
            gt_values.append(i_gt)
            alt_values.append(i_alt)
            fid_values.append(fid)
    
                
            alt = alt.transpose(1, 2).to(config.device)

            output, _, _ = model(alt, fid)
            output = output.squeeze()
            fixed_values.append(output)
            mae_output.append(torch.mean(torch.abs(i_gt - output)).cpu().numpy())

    # create kde plots
    # altered vs ground truth
    gt_values = torch.cat(gt_values).cpu().numpy()
    alt_values = torch.cat(alt_values).cpu().numpy()
    fixed_values = torch.cat(fixed_values).cpu().numpy()
    fid_values = torch.cat(fid_values).cpu().numpy()

    flights = {}
    for i in fid_values:
        if i not in flights:
            flights[i] = 0
        else:
            flights[i] += 1

    # create bar chart for fid values
    plt.bar(np.arange(len(flights.keys())), flights.values())
    plt.xticks(np.arange(len(flights.keys())), flights.keys())        
    plt.savefig(results_path / "fid_distribution.png")

    print(f"Alt values: {len(alt_values)}")
    print(f"GT Values: {len(gt_values)}")
    print(f"Fixed values: {len(fixed_values)}")

    # total measurement
    total_mae_output = np.mean(np.array(mae_output))

    # altered vs ground truth
    create_kde(gt_values,
               alt_values,
               "ground truth",
               "altered values",
               results_path / "kde_alt_vs_gt.png")
    
    # fixed vs ground truth
    create_kde(gt_values,
               fixed_values,
               "ground truth",
               "fixed values",
               results_path / "kde_evaluation.png",
               text=f"MAE: {total_mae_output:.5f}")
    
    
    print(f"MAE: {total_mae_output}")
    print(f"Finished in {time.time() - start_time} seconds")
