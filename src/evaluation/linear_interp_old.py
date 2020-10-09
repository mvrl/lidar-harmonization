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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm
from pptk import kdtree, viewer
from src.dataset.tools.metrics import create_kde

from src.training.forward_pass import forward_pass, get_metrics
from src.dataset.tools.dataloaders import get_dataloader
from scipy.interpolate import griddata
from scipy.stats import gaussian_kde
from sklearn.neural_network import MLPRegressor


def generate_map(neighborhood_size=5):

    start_time = time.time() 
    batch_size = 50
    target_scan = 1

    print("generating fix for big_tile dataset")
    results_path = Path(f"results/linear_interpolation")
    results_path.mkdir(parents=True, exist_ok=True)

    method = "MLP"   # "lstsq" "MLP"
    print(f"Using method: {method}")

    # get training data to fit least squares 
    train_csv = Path("dataset/150/train.csv")
    train_dataloader = get_dataloader(train_csv, batch_size, 6)

    # prep for tile creation
    tile_path = results_path / "tile"
    tile_path.mkdir(parents=True, exist_ok=True)

    tile_csv = Path("dataset/big_tile_no_overlap/big_tile_dataset.csv")
    tile_dataloader = get_dataloader(tile_csv, batch_size, 6, drop_last=False)


    # In this step, we interpolate the center point for each training example.
    #   This builds the dataset for harmonization by providing a 1-1 
    #   correspondence between known points from the target scan with the newly 
    #   interpolated points from the source scan

    running_loss = 0
    with torch.no_grad():
        dataset = np.empty((0, 5))
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch_idx, batch in enumerate(pbar):
            # we only care about learning some relationship between h and i
            data, h_target, i_target = batch

            # given a neighborhood around h_target, interpolate the intensity
            #   value at this location. Idx 0 is harmonized value, idx 1 is 
            #   "true" intensity at this location for the source flight 
            #   (i_target). idx 2-151 is the neighborhood
            interpolation = np.concatenate([
                    griddata(
                        n[2:,:2],
                        n[2:,3], 
                        n[0,:2]) for n in data])
            
            curr_source_scan_num = data[:, 1, 8]
            curr_target_scan_num = data[:, 0, 8]

            new_batch = np.concatenate((
                interpolation.reshape(-1, 1), 
                i_target.numpy().reshape(-1, 1), 
                h_target.numpy().reshape(-1, 1),
                curr_source_scan_num.numpy().reshape(-1, 1),
                curr_target_scan_num.numpy().reshape(-1, 1)
                ), axis=1)


            # looks like we get some bad values occassionally
            nans = np.where(np.isnan(interpolation))
            new_batch = np.delete(new_batch, nans, axis=0)
            
            loss = np.mean(np.abs(new_batch[:, 0] - new_batch[:, 1]))
            
            running_loss += loss * batch_size
            total_loss = running_loss / (((batch_idx+1) * batch_size))
            pbar.set_postfix({
                "icurr": f"{float(loss):.3f}", 
                "itotal": f"{float(total_loss):.3f}"})
            
            dataset = np.concatenate((dataset, new_batch))

            # if len(dataset) >= 10000:
            #     break

    print(f"Total Interpolation Loss: {running_loss/len(train_dataloader)}")
    
    code.interact(local=locals())

    ## Regress harmonization
    
    # filter dataset on source-target == source-1
    dataset_f = dataset[dataset[:, 4] == 1]

    # create transform table
    sources = np.unique(dataset[:, 3])
    transforms = {}

    for s in sources:
        # filter on source -- each target-source pair needs a transformation
        dataset_f_s = dataset_f[dataset_f[:, 3] == s]
        X = np.expand_dims(dataset_f_s[:, 0], 1)
        y = dataset_f_s[:, 2]
        
        # transforms now holds a lstsq's solution for each source-target pair
        if method is "lstsq":
            transforms[(int(s), 1)] = np.concatenate(
                np.linalg.lstsq(X, y, rcond=None)[:2])

        elif method is "MLP":
            transforms[(int(s), 1)] = MLPRegressor(
                (100, 100), random_state=1, max_iter=300).fit(X, y)

        else:
            exit(f"No method defined... {method}")

    running_loss = 0
    with torch.no_grad():
        fixed_tile = np.empty((0, 10), dtype=np.float64)
        pbar = tqdm(tile_dataloader, total=len(tile_dataloader))
        for batch_idx, batch in enumerate(pbar):
            data, h_target, i_target = batch
            # tile_data = data[:, 0, :].numpy().copy()  # harmonized gt
            tile_data = data[:, 1, :].numpy().copy()    # alt gt == i_target
            data = data.numpy()
            
            # in this section, the center intensity is given, since the first
            # member of the neighborhood is the center point. We only need to 
            # apply the transformation calculated above
            
            # Get data for prediction
            intensity = tile_data[:, 3].reshape(-1, 1)

            source_scan = int(data[0, 0, 8])  # just pick any; all same
            t = transforms[(source_scan, target_scan)]
            # Transform intensities
            if method is "lstsq":
                new_intensity = np.sum(intensity * t, axis=1).reshape(-1, 1)

            if method is "MLP":
                new_intensity = t.predict(intensity).reshape(-1, 1)
            
            tile_data = np.concatenate((
                tile_data[:, :4], 
                new_intensity,
                tile_data[:, 4:]), axis=1)
            
            loss = np.mean(np.abs(new_intensity - intensity))
            
            running_loss += loss * batch_size
            total_loss = running_loss / (((batch_idx+1) * batch_size))
            pbar.set_postfix({
                "hcur": f"{float(loss):.3f}", 
                "htot": f"{float(total_loss):.3f}"})

            fixed_tile = np.concatenate((fixed_tile, tile_data))

            # if len(fixed_tile) >= 100000:
            #     break

    print(f"Total harmonization loss: {running_loss/len(tile_dataloader)}")
    
    # Save tile to visualize later
    np.savetxt(tile_csv.parents[0] / f"fixed_li_{method}.txt.gz", fixed_tile)  

    code.interact(local=locals())

    # make a nice figure for interpolation and harmonization accuracies
    s1 = np.random.choice(len(i_target), size=5000)
    i_target = dataset[:, 1][s1]
    i_preds = dataset[:, 0][s1]
    i_mae = np.mean(np.abs(i_target - i_preds))

    s2 = np.random.choice(len(h_target), size=5000)
    h_target = fixed_tile[:, 3][s2]
    h_preds = fixed_tile[:, 4][s2]
    h_mae = np.mean(np.abs(h_target - h_preds))

    ixy = np.vstack([i_preds, i_target])
    iz = gaussian_kde(ixy)(ixy)
    iidx = iz.argsort()
    i_target, i_preds, iz = i_target[iidx], i_preds[iidx], iz[iidx]

    hxy = np.vstack([h_preds, h_target])
    hz = gaussian_kde(hxy)(hxy)
    hidx = hz.argsort()
    h_arget, h_preds, hz = h_target[hidx], h_preds[hidx], hz[hidx]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f"Linear Interpolation with {method} Harmonization")
    fig.set_size_inches(13, 5)

    ax1.set_title("Harmonization Predictions vs GT")
    ax1.scatter(h_target, h_preds, s=7, c=hz)
    ax1.plot([0,1], [0,1])
    ax1.margins(x=0)
    ax1.margins(y=0)
    ax1.set_xlabel("Ground Truth")
    ax1.set_ylabel("Predictions")
    ax1.text(.5, 0, str(h_mae))

    ax2.set_title("Interpolation Predictions vs GT")
    ax2.scatter(i_target, i_preds, s=7, c=iz)
    ax2.plot([0, 1], [0, 1])
    ax2.margins(x=0)
    ax2.margins(y=0)
    ax2.set_xlabel("Ground Truth")
    ax2.set_ylabel("Predictions")
    ax2.text(.5, 0, str(i_mae))

    plt.tight_layout()
    plt.savefig(results_path / f"qual_li_{method}.png")
    plt.close()

            

if __name__ == "__main__":
    generate_map()
