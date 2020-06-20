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
from util.transforms import CloudCenter, ToTensor, CloudIntensityNormalize
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
from dataset.lidar_dataset import BigMapDataset

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

    transform = Compose([
        CloudCenter(), 
        CloudIntensityNormalize(512), 
        ToTensor()])
    
    big_tile_alt_ = np.load("dataset/big_tile/big_tile_alt.npy")
    big_tile_gt_ = np.load("dataset/big_tile/big_tile_gt.npy")
    
    sample = np.random.choice(len(big_tile_gt_[:, 3]), size=5000)
    create_kde(
            big_tile_gt_[sample][:, 3],
            big_tile_alt_[sample][:, 3],
            "pre ground truth",
            "pre altered values",
            "results/big_tile/gen_map_kde_pre_alt_vs_gt.png")
    print("Saved pre-generation measured response curve")  # check
    kd = kdtree._build(big_tile_alt_[:, :3])  # build tree

    # query every point for its 10 nearest neighbors
    # Note: we add one to the neighborhood size since 
    #   the query point will be included in the query
    query = kdtree._query(
            kd, 
            big_tile_alt_[:, :3], 
            k=neighborhood_size)

    # only some points have 10 neighbors within dmax, lowering 
    # dmax may increase accuracy of qualitative evaluation, but
    # less points will be evaluated overall. 
    my_query = []
    for i in query:
        if len(i) == 10:
            my_query.append(i)
    
    good_sample_ratio = ((len(query) - len(my_query))/len(query)) * 100
    print(f"Found {good_sample_ratio} percent of points with not enough close neighbors!")
    query = my_query

    # get neighborhoods
    big_tile_alt = big_tile_alt_[query]
    big_tile_gt = big_tile_gt_[query]

    # realign gt and alt with query
    big_tile_alt_ = big_tile_alt[:, 0, :]
    big_tile_gt_ = big_tile_gt[:, 0, :]

   
    dataset = BigMapDataset(big_tile_gt, big_tile_alt, transform=transform)
    dataloader = DataLoader(
            dataset,
            batch_size=1000,
            num_workers=config.num_workers)

    fix_values = []
    alt_values = []
    gt_values = []
    mae_values = []
    print(f"starting inference for tileset of length {len(dataset)}")
    print(f"batch_size = 1000")
    print(f"number of iterations = {len(dataset)/1000}")
    max_batch = len(dataset)//1000
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            gt, alt = batch

            # get center point true intensity
            i_gt = gt[:, 0, 3]
            gt_values.append(i_gt)
            
            # extract pt_src_id information
            fid = alt[:, :, 8][:, 0].long().to(config.device)

            # remove pt_src_id column
            # centerpoint stays as it is a real test value
            alt_values.append(alt[:, 0, 3])
            alt = alt[:, 1:, :8]
            
            # put channels first
            alt = alt.transpose(1, 2).to(config.device)
            
            output, _, _ = model(alt, fid)
            output = torch.clamp(output.squeeze(), 0, 1)
            
            mae = torch.mean(torch.abs(output - i_gt.to(config.device))).item()
            mae_values.append(mae)
            print(f"[{batch_idx}/{max_batch}: MAE: {mae}")
            fix_values.append(output)
            
    # create kde plots
    
    count = len(alt_values)
    indices = np.arange(count)
    my_indices = np.random.choice(indices, size=5000)

    alt_values = torch.cat(alt_values).cpu().numpy()
    gt_values = torch.cat(gt_values).cpu().numpy()
    fix_values = torch.cat(fix_values).cpu().numpy() * 512

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

    
    big_tile_alt_fixed = big_tile_alt_.copy()
    big_tile_alt_fixed[:, 3] = fix_values
    np.save("dataset/big_tile/big_tile_alt_fixed.npy", big_tile_alt_fixed)
    print("Saved the fixed point cloud!")
    
    # Generate fixed version
    print("Generating view...")
    base_flight_tile = np.load("dataset/big_tile/base_flight_tile.npy")

    # spatial dimensions
    tile = np.concatenate((base_flight_tile, big_tile_gt_))
    v = viewer(tile[:, :3])

    # attributes
    attr0 = tile[:, 8]  # flight numbers
    attr1 = tile[:, 3]  # ground truth
    attr2 = np.concatenate((base_flight_tile[:, 3], big_tile_alt_[:, 3])) # alt 
    attr3 = np.concatenate((base_flight_tile[:, 3], big_tile_alt_fixed[:, 3])) # fix

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
    v.close()
