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
    big_tile_gt = np.load("dataset/big_tile/big_tile_gt.npy")
    
    kd = kdtree._build(big_tile_alt_[:, :3])  # build tree

    # query every point for its 10 nearest neighbors
    # Note: we add one to the neighborhood size since 
    #   the query point will be included in the query
    query = kdtree._query(
            kd, 
            big_tile_alt_[:, :3], 
            k=neighborhood_size)

    big_tile_alt = big_tile_alt_[query]
    big_tile_gt = big_tile_gt[query]
    
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
            
    
    code.interact(local=locals())
    # create kde plots
    
    count = len(alt_values)
    indices = np.arange(count)
    my_indices = np.random.choice(indices, size=5000)

    alt_values = torch.cat(alt_values).cpu().numpy()[my_indices]
    gt_values = torch.cat(gt_values).cpu().numpy()[my_indices]
    fix_values = torch.cat(fix_values).cpu().numpy()[my_indices]

    final_mae = torch.mean(torch.tensor(mae_values)).item()

    # altered vs gt
    create_kde(gt_values,
            alt_values,
            "ground truth",
            "altered values",
            "results/big_tile/gen_map_kde_alt_vs_gt.png")

    # fixed vs gt
    create_kde(gt_values,
            fix_values,
            "ground truth",
            "fixed values",
            "results/big_tile/gen_map_kde_evaluation.png",
            text=f"MAE: {final_mae}")

    

    code.interact(local=locals())
    fix_values = torch.cat(fix_values)
    big_tile_alt_fixed = big_tile_alt_.copy()
    big_tile_alt_fixed[:, 3] = fix_values.cpu().numpy()
    np.save("dataset/big_tile/big_tile_alt_fixed.npy", big_tile_alt_fixed)


    
