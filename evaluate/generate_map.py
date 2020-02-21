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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import gaussian_kde

from pptk import kdtree, viewer
from laspy.file import File
from util.apply_rf import apply_rf

from model import IntensityNet
from dataset.lidar_dataset import BigMapDataset


def generate_map(config=None, state_dict=None, tile_directory=None, tree_size=50, viewer_sample=3, **kwargs):

    start_time = time.time()

    # get flight path files:
    laz_files_path = Path(r"dataset/dublin_flights")
    laz_files = [file for file in laz_files_path.glob('*.npy')]
    file_count = len(laz_files)
    
    # Load model to be evaulated
    model = IntensityNet(num_classes=config.num_classes).to(config.device).double()
    model.load_state_dict(torch.load(state_dict))
    model.eval()

    # Load response functions
    with open('dataset/response_functions.json') as json_file:
        data = json.load(json_file)

    # Evaluate the big tile first
    big_tile = np.load("dataset/big_tile/big_tile.npy")
    big_tile_alt = np.load("dataset/big_tile/big_tile_alt.npy")

    # load flight 1, create an altered version using flight 4 transform:
    f1 = np.load(laz_files[0])
    f1_altered = apply_rf("dataset/response_functions.json", f1, 4)

    # build kdtree, query for neighborhoods. k must equal value in original value
    print("building kdtree...")
    kd = kdtree._build(f1[:, :3])
    query = kdtree._query(kd, big_tile[:,:3], k=50)

    clouds = f1_altered[query]
    flight_ids = np.full((len(clouds),1), 4).squeeze()

    transforms = Compose([CloudNormalizeBD(), ToTensorBD()])
    dataset = BigMapDataset(clouds, flight_ids, transform=transforms)

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers)

    fixed_intensities = []
    print("Fixing point cloud")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            clouds, flight_ids = batch
            flight_ids = flight_ids.long().to(config.device)
            clouds = clouds.transpose(1,2).to(config.device)
            
            output, _, _ = model(clouds, flight_ids)
            fixed_intensities.append(output)    

    fixed_intensities = torch.cat(fixed_intensities)
    fixed_intensities = fixed_intensities.cpu().numpy()  # get as ndarray

    original_intensities = big_tile[:, 3]
    altered_intensities = big_tile_alt[:, 3]

    v = viewer(big_tile[:,:3])
    v.set(point_size=0.005)
    v.attributes(original_intensities, altered_intensities, fixed_intensities)
    code.interact(local=locals())

