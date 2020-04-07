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
from torchvision.transforms import Compose
from util.transforms import CloudNormalizeBD, ToTensorBD
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
                **kwargs):

    print(f"measuring accuracy on {dataset_csv}")

    dataset = LidarDataset(dataset_csv, transform=transforms)

    suffix = dataset_csv.split("/")[1]
    if suffix.split("_")[0] == '0':
        print("Entering zero-neighbors mode")
        zero_neighbors = True
    else:
        zero_neighbors = False
    dataloader = DataLoader(
        dataset,
        batch_size=100,
        num_workers=config.num_workers,
        drop_last=True)

    mae_output = []
    alt_values = []
    fixed_values = []
    gt_values = []
    model = IntensityNet(num_classes=config.num_classes).to(config.device).double()
    model.load_state_dict(torch.load(state_dict))
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            eval_time = time.time()
            gt, alt, fid, _ = batch
            
            # Get the intensity of the ground truth point
            i_gt = gt[:,0,3].to(config.device)
            fid = fid.to(config.device)

            alt_values.append(alt[:, 0, 3])
            gt_values.append(i_gt)
            
            alt = alt[:, 0:, :]
           
            alt = alt.transpose(1, 2).to(config.device)

            output, _, _ = model(alt, fid)
            output = output.squeeze()
            fixed_values.append(output)
            mae_output.append(torch.mean(torch.abs(i_gt - output)).cpu().numpy())

    # create kde plots
    # altered vs ground truth
    alt_values = np.stack([i.cpu().numpy() for i in alt_values]).flatten()
    gt_values = np.stack([i.cpu().numpy() for i in gt_values]).flatten()
    fixed_values = np.stack([i.cpu().numpy() for i in fixed_values]).flatten()
    print(f"Alt values: {len(alt_values)}")
    print(f"GT Values: {len(gt_values)}")
    print(f"Fixed values: {len(fixed_values)}")

    # altered vs ground truth
    create_kde(gt_values,
               alt_values,
               "ground truth",
               "altered values",
               f"results/{suffix}/kde_alt_vs_gt.png")
    
    # fixed vs ground truth
    create_kde(gt_values,
               fixed_values,
               "ground truth",
               "fixed values",
               f"results/{suffix}/kde_evaluation.png")
    
    # total measurement
    total_mae_output = np.mean(np.array(mae_output))

    print(f"MAE: {total_mae_output}")
    
