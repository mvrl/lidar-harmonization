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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import gaussian_kde

from pptk import kdtree, viewer
from laspy.file import File

from model import IntensityNet

def evaluate(state_dict=None, dataset=None, tree_size=50, viewer_sample=3, **kwargs):

    start_time = time.time()

    # get flight path files:
    laz_files_path = Path(r"dataset/dublin_flights")
    laz_files = [file for file in laz_files_path.glob('*.npy')]
    file_count = len(laz_files)
    
    # Load model to be evaulated
    model = IntensityNet().double().cuda()
    model.load_state_dict(torch.load(state_dict))
    model.eval()

    # Sort dataset by flightpath to reduce redundant loading
    dataset.df = dataset.df.sort_values(by=['flight_path_file'])

    # Load response functions
    with open('dataset/response_functions.json') as json_file:
        data = json.load(json_file)

    curr_fl_path = None
    fixed_vs_gt = []
    fixed_vs_gt_mae = []
    alt_vs_gt_mae = []
    random_viewer_sample = np.random.randint(0,
                                             high=len(dataset),
                                             size=viewer_sample)

    # Evaluate the big tile first
    big_tile = np.load("dataset/big_tile/big_tile.npy")
    big_tile_alt = np.load("dataset/big_tile/big_tile_alt.npy")



    # need to generate a 30000 x neighbor from big_tile

    # load flight 1:
    f1 = np.load(laz_files[0])
    f1_alt = f1.copy()
    f1_alt_i = f1_alt[:, 3]
    f1_alt_i = np.interp(f1_alt_i, np.array(data['brightness']['2'])*255, np.array(data['intensities']['2'])*255)
    f1_alt[:, 3] = f1_alt_i

    
    kd = kdtree._build(f1[:, :3])

    query = kdtree._query(kd, big_tile[:,:3], k=50)
    with torch.no_grad():
        n = f1_alt[query]
        n = torch.tensor(n).cuda()
        n = n.transpose(1,2)
        f = torch.full((30000,1), 2).long().cuda().squeeze()
    
        results = []
        for i in range(0, 30000, 100):
            output,_,_ = model(n[i:i+100], f[i:i+100])
            results.append(output.cpu())
    
    

        code.interact(local=locals())
    exit()
    for i in range(len(dataset)):
        # iterate over dataset
        gt, alt, flight_num, fl_path = dataset[i]

        # construct the altered flight if we haven't seen this flightpath already
        if curr_fl_path is not fl_path:
            curr_fl_path = fl_path
            print(f"Loading {fl_path}...")
            flight = np.load(fl_path[:-3] + 'npy')

            flight_i = flight[:, 3]
            flight_ai = np.interp(flight_i,
                              np.array(data['brightness'][str(flight_num)])*255,
                              np.array(data['intensities'][str(flight_num)])*255)

            flight_a = np.copy(flight)
            flight_a[:, 3] = flight_ai  # altered flight
            
            # build the kdtree on the altered flight
            kd = kdtree._build(flight_a)
        else:
            print(f"{fl_path} already loaded -> ({flight.shape})")

        # query the kdtree
        # code.interact(local=locals())
        alt = alt[1:, :] # chop out f1 point
        gt = gt[1:,:]
        flight_num = torch.tensor(flight_num).cuda()
        query = kdtree._query(kd, alt.numpy(), k=tree_size)

        # index the altered flight to return clouds
        clouds = flight_a[query]  # tree_size, tree_size, 4

        # these clouds must be centered before inference
        clouds[:,:,:3] = np.stack([cloud[:,:3] - cloud[0][:3] for cloud in clouds])
        clouds = torch.tensor(clouds).cuda()

        # perform inference - get "fixed intensities"
        code.interact(local=locals())
        pred = torch.stack([model(cloud.unsqueeze(2), flight_num)[0] for cloud in clouds])

        # After inference, can re-make alt with these new intensites. 
        fixed_sample = np.copy(alt)
        fixed_sample[:,3] = pred[0, :].cpu().detach().numpy()

        # code.interact(local=locals())

        # Display fixed sample and alt side by side, only 3 times @ rand?
        if i: # in random_viewer_sample:
            # display the fixed sample, gt, alt
            disp_alt = alt.numpy()
            disp_alt[:,1] = disp_alt[:,1]+2
            
            disp_gt = gt.numpy()
            disp_gt[:,1] = disp_gt[:,1]-2
            
            disp = np.concatenate((
                fixed_sample,
                disp_alt,
                disp_gt), axis=0)
            print("starting viewer...")
            
            v = viewer(disp[:,:3])
            v.attributes(disp[:,3])
            v.set(point_size=0.02,
                  show_axis=False,
                  bg_color=[1,1,1,1],
                  show_grid=False)
        
            input("Press Enter to Continue")
            v.close()

        # prep KDE plot for fixed vs gt
        alt = alt.numpy()
        gt = gt.numpy()
        fixed_vs_gt.append(np.array([fixed_sample[:,3], gt[:,3]]))
        fixed_vs_gt_mae.append(np.mean(np.absolute(alt[:,3]-gt[:,3]), axis=0))

        # Report difference between altered and gt...? MAE?
        alt_vs_gt_mae.append(np.mean(np.absolute(alt[:,3] - gt[:,3])))

    # Create KDE Plot -- this takes a while, maybe sample this to avoid using
    # the whole thing?
    print("Generating KDE for visualization")
    # code.interact(local=locals())
    fixed_vs_gt = np.concatenate(fixed_vs_gt, axis=1)
    z = gaussian_kde(fixed_vs_gt)(fixed_vs_gt)
    fig, ax = plt.subplots()
    ax.scatter(fixed_vs_gt[0], fixed_vs_gt[1])
    plt.title("Predicted vs Ground Truth")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.savefig("evaluation/kde_eval.png")

    print(f"MAE for altered tiles vs the ground truth: {np.mean(alt_vs_gt_mae)}")
    print(f"MAE for fixed tiles vs the ground truth: {np.mean(fixed_vs_gt_mae)}")
    print(f"finished in {time.time() - start_time}")

