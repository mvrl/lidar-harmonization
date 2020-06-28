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
from util.transforms import CloudCenter, ToTensor, CloudIntensityNormalize, LoadNP
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import gaussian_kde

from pptk import kdtree, viewer


from util.metrics import create_kde

from model import IntensityNet
from dataset.lidar_dataset import LidarDataset

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def generate_map(config=None,
                 state_dict=None,
                 tileset_directory=None,
                 viewer_sample=3,
                 neighborhood_size=None,
                 **kwargs):

    start_time = time.time() 
    model = IntensityNet(num_classes=config.num_classes).to(config.device).double()
    model.load_state_dict(torch.load(state_dict))
    model.eval()

    batch_size = 1000
    transform = Compose([
        LoadNP(),
        CloudCenter(), 
        CloudIntensityNormalize(512), 
        ToTensor()])
    
    dataset = LidarDataset("dataset/big_tile/big_tile_dataset.csv", transform=transform)
    
    dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=config.num_workers)

    fix_values = []
    fix_tile = []
    alt_values = []
    gt_values = []
    mae_values = []
    fid_values = []

    print(f"starting inference for tileset of length {len(dataset)}")
    print(f"batch_size = {batch_size}")
    print(f"number of iterations = {len(dataset)/batch_size}")
    max_batch = len(dataset)//batch_size
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            gt, alt = batch
            sample = np.random.choice(len(gt), size=batch_size//4)

            # get center point true intensity
            i_gt = gt[:, 0, 3]
            gt_values.append(i_gt[sample])
            
            # extract pt_src_id information
            fid = alt[:, :, 8][:, 0]
            fid_values.append(fid[sample])
            fid = fid.long().to(config.device)

            # centerpoint stays for qualitative evaluation as it is a real value
            alt_values.append(alt[:, 0, 3][sample])
            alt = alt[:, 0:, :8]
            
            # put channels first
            alt = alt.transpose(1, 2).to(config.device)
            
            output, _, _ = model(alt, fid)
            output = torch.clamp(output.squeeze(), 0, 1)
            
            mae = torch.mean(torch.abs(output - i_gt.to(config.device))).item()
            mae_values.append(mae)
            print(f"[{batch_idx}/{max_batch}: MAE: {mae}")
            fix = gt.clone()[:, 0, :]
            fix[:, 3] = output
            fix_values.append(output[sample])
            fix_tile.append(fix)
            
    # create kde plots
    
    count = len(alt_values)
    indices = np.arange(count)
    my_indices = np.random.choice(indices, size=5000)

    alt_values = torch.cat(alt_values).cpu().numpy()
    gt_values = torch.cat(gt_values).cpu().numpy()
    fix_values = torch.cat(fix_values).cpu().numpy()
    fid_values = torch.cat(fid_values).cpu().numpy()

    final_mae = torch.mean(torch.tensor(mae_values)).item()

    # altered vs gt
    create_kde(gt_values[my_indices],
            alt_values[my_indices],
            "ground truth",
            "altered values",
            "results/big_tile/gen_map_kde_alt_vs_gt.png")

    # fixed vs gt
    create_kde(gt_values[my_indices],
            fix_values[my_indices],
            "ground truth",
            "fixed values",
            "results/big_tile/gen_map_kde_evaluation.png",
            text=f"MAE: {final_mae}")

    # bar plot for fid values
    flights = {}
    for i in fid_values:
        if i not in flights:
            flights[i] = 0
        else:
            flights[i] += 1

    plt.bar(np.arange(len(flights.keys())), flights.values())
    plt.xticks(np.arange(len(flights.keys())), flights.keys())
    plt.savefig("results/big_tile/fid_distribution.png")

    # load in tile assets
    big_tile_gt = np.load("dataset/big_tile/big_tile_gt.npy")
    big_tile_alt = np.load("dataset/big_tile/big_tile_alt.npy")
    big_tile_fix = torch.cat(fix_tile).cpu().numpy()
    
    big_tile_fix[:, 3] *= 512 # scale intensity to 0-512
    np.save("dataset/big_tile/big_tile_alt_fixed.npy", big_tile_fix)
    print("Saved the fixed point cloud!")
    
    # Generate fixed version
    print("Generating view...")
    base_flight_tile = np.load("dataset/big_tile/base_flight_tile.npy")

    # spatial dimensions
    tile = np.concatenate((base_flight_tile, big_tile_gt))
    v = viewer(tile[:, :3])

    # attributes
    attr0 = tile[:, 8]  # flight numbers
    attr1 = tile[:, 3]  # ground truth
    attr2 = np.concatenate((base_flight_tile[:, 3], big_tile_alt[:, 3])) # alt 
    attr3 = np.concatenate((base_flight_tile[:, 3], big_tile_fix[:, 3])) # fix

    v.attributes(attr0, attr1, attr2, attr3)
    v.set(bg_color=(1,1,1,1), show_axis=False, show_grid=False, show_info=False)
    v.set(lookat=(316330.094, 234122.562, 6.368))
    v.set(phi=-1.57079637, theta=1.57079637, r=170.17500305)
    v.capture("results/big_tile/0_flights.png")
    v.set(curr_attribute_id=1)
    v.capture("results/big_tile/3_gt_comparison.png")
    v.set(curr_attribute_id=2)
    v.capture("results/big_tile/1_alt_comparison.png")
    v.set(curr_attribute_id=3)
    v.capture("results/big_tile/2_fixed_comparison.png")
    code.interact(local=locals())
    v.close()
