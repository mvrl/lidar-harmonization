import os
import json
import time
import code
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from src.datasets.tools.metrics import create_kde

from src.datasets.tools.dataloaders import get_transforms
from src.datasets.tools.metrics import create_interpolation_harmonization_plot
from src.harmonization.inet_pn1 import IntensityNet


def dl_interp_model(model, dataloader, config):
    target_camera = int(config['dataset']['target_scan'])
    n_size = config['train']['neighborhood_size']
    dataset_csv = Path(config['dataset']['eval_save_path']) / "dataset.csv"

    results_path = Path(config['dataset']['eval_save_path']) 
    print(f"Saving charts to {results_path}")
    results_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        fixed_tile = np.empty((0, 11), dtype=np.float64)
        pbar = tqdm(dataloader, total=len(dataloader))
        for batch_idx, batch in enumerate(pbar):
            

            # store the center point for each neighborhood. Note that the 
            # harmonized gt has been stripped out but is stored in h_target
            # the center point below actually has i_target in channel 4
            tile_data = batch[:, 0, :].numpy()
            h_target = batch[:, 0, 3].clone()
            i_target = batch[:, 1, 3].clone()

            batch = batch.to(device)
            # specify that we want to harmonize to `target_camera` by 
            # overwriitng the original target camera value
            batch[:, 0, -1] = target_camera

            harmonization, interpolation, _  = model(batch)
            
            tile_data = np.concatenate(
                (
                    tile_data[:, :3],                # XYZ,
                    h_target.cpu().numpy().reshape(-1, 1), # h_target (3)
                    harmonization.cpu().numpy(),     # h_pred   (4)
                    i_target.cpu().numpy().reshape(-1, 1), # i_target (5)
                    interpolation.cpu().numpy(),     # i_pred   (6)
                    tile_data[:, 5:]                 # everything else
                ), 
                axis=1)       

            fixed_tile = np.concatenate((fixed_tile, tile_data))

    h_mae, i_mae = create_interpolation_harmonization_plot(
            fixed_tile[:, 3],  
            np.clip(fixed_tile[:, 4],0,1),
            np.mean(np.abs(fixed_tile[:, 3] - fixed_tile[:, 4])),
            fixed_tile[:, 5],
            np.clip(fixed_tile[:, 6],0,1),
            np.mean(np.abs(fixed_tile[:, 4] - fixed_tile[:, 5])),
            "qual", results_path / f"qual_kde_{n_size}_dl.png")


    np.savetxt(results_path / f"fixed_dl_{n_size}.txt.gz", fixed_tile)

    return h_mae, i_mae


def dl_interp(state_dict, target_camera=1):

    print("generating fix for big_tile dataset")

    state_dict = Path(state_dict)
    n_size = int(state_dict.stem.split("_")[0])
    print(f"neighborhood size: {n_size}")
    
    if "shift" in str(state_dict):
        print("Running on shifted dataset model")
        dataset_csv = Path("dataset/synth_crptn+shift/big_tile_no_overlap/big_tile_dataset.csv")

    else:
        print("Running on default dataset model")
        dataset_csv = Path("datasets/dublin/old/synth_crptn/big_tile_no_overlap/big_tile_dataset.csv")

    results_path = state_dict.parents[0]
    print(f"Saving charts to {results_path}")
    results_path.mkdir(parents=True, exist_ok=True)

    dataloader = get_dataloader(dataset_csv, 50, True, 8, drop_last=False)

    device = torch.device("cuda:0")
    model = IntensityNetPN1(neighborhood_size=n_size).double().to(device)
    model.load_state_dict(torch.load(state_dict))
    model.eval()

    with torch.no_grad():
        fixed_tile = np.empty((0, 11), dtype=np.float64)
        pbar = tqdm(dataloader, total=len(dataloader))
        for batch_idx, batch in enumerate(pbar):
            data, h_target, i_target = batch

            # store the center point for each neighborhood. Note that the 
            # harmonized gt has been stripped out but is stored in h_target
            # the center point below actually has i_target in channel 4
            tile_data = data[:, 0, :].numpy()
            data = data.to(device)

            # specify that we want to harmonize to `target_camera`
            data[:, 0, -1] = target_camera

            harmonization, interpolation, _ = model(data)
            
            tile_data = np.concatenate(
                (
                    tile_data[:, :3],                # XYZ,
                    h_target.numpy().reshape(-1, 1), # h_target (3)
                    harmonization.cpu().numpy(),     # h_pred   (4)
                    i_target.numpy().reshape(-1, 1), # i_target (5)
                    interpolation.cpu().numpy(),     # i_pred   (6)
                    tile_data[:, 5:]                 # everything else
                ), 
                axis=1)       

            fixed_tile = np.concatenate((fixed_tile, tile_data))

    create_interpolation_harmonization_plot(
            fixed_tile[:, 3],  
            np.clip(fixed_tile[:, 4],0,1),
            np.mean(np.abs(fixed_tile[:, 3] - fixed_tile[:, 4])),
            fixed_tile[:, 5],
            np.clip(fixed_tile[:, 6],0,1),
            np.mean(np.abs(fixed_tile[:, 4] - fixed_tile[:, 5])),
            "qual", results_path / f"qual_kde_{n_size}_dl.png")


    np.savetxt(dataset_csv.parents[0] / f"fixed_dl_{n_size}.txt.gz", fixed_tile)


if __name__ == "__main__":
    dl_interp("results/5/5_epoch=21.pt")
