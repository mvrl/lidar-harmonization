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
matplotlib.use('Agg')

from tqdm import tqdm
from pptk import kdtree, viewer
from src.dataset.tools.metrics import create_kde

from src.training.forward_pass import forward_pass, get_metrics
from src.dataset.tools.dataloaders import get_dataloader
from src.dataset.tools.metrics import create_interpolation_harmonization_plot
from src.harmonization.model_npl import HarmonizationNet


def generate_map(neighborhood_size=5):

    start_time = time.time() 
    print("generating fix for big_tile dataset")
    results_path = Path(f"results/{neighborhood_size}")
    results_path.mkdir(parents=True, exist_ok=True)

    # prep for tile creation
    tile_path = results_path / "tile"
    tile_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")

    fixed_values = []
    fixed_tile = []
    alt_values = []
    gt_values = []
    mae_values = []
    fid_values = []

    model = HarmonizationNet(neighborhood_size=neighborhood_size).double().to(device)
    model.load_state_dict(torch.load("results/5/5_epoch=13.pt"))
    model.eval()

    dataset_csv = "dataset/big_tile_no_overlap/big_tile_dataset.csv"
    dataloader = get_dataloader(dataset_csv, 50, 4, drop_last=False)
    my_data = []
    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader))
        for batch_idx, batch in enumerate(pbar):
            with torch.set_grad_enabled(False):
                data, h_target, i_target = batch
                data = data.to(device)
                h_target = h_target.to(device)
                i_target = i_target.to(device)
                harmonization, interpolation = model(data)
                r = {
                        'loss': None,
                        'metrics' : {
                            'h_target': h_target.detach().cpu(),
                            'harmonization': harmonization.squeeze().detach().cpu(),
                            'i_target': i_target.detach().cpu(),
                            'interpolation': interpolation.squeeze().detach().cpu()}}
                my_data.append(r)

    h_target, h_preds, h_mae, i_target, i_preds, i_mae = get_metrics(my_data)
    create_interpolation_harmonization_plot(
            h_target, h_preds, h_mae,
            i_target, i_preds, i_mae,
            "qual", results_path / f"qual_kde_{neighborhood_size}.png")
                        

    

    

            
            
"""
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
    v.close()"""

if __name__ == "__main__":
    generate_map()
