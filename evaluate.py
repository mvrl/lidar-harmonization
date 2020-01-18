import os
import json
import code
from pathlib import Path
from datetime import datetime as dt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pptk import kdtree, viewer
from laspy.file import File


from model import IntensityNet

def evaluate(state_dict, dataset, tree_size=50, viewer_sample=0):
    """ 
    This function runs evaluation for the intensity prediction on a given model.
    
    Needs to evaluate the altered tiles: 
        Do this by loading the flight for each tile, then grab the cloud around 
        each point in the tile.

        Compare this with the ground truth tile.

        Additionally, record the original amount of error between the ground 
        truth tile and the altered tile
    """

    # Load model to be evaulated
    model = IntensityNet().double().cuda()
    model.load_state_dict(torch.load(state_dict))
    model.eval()

    # Sort dataset by flightpath to reduce redundant loading
    dataset.df = dataset.df.sort_values(by=['flight_path_file'])

    # Load response functions
    with open('response_functions.json') as json_file:
        data = json.load(json_file)

    curr_fl_path = None
    fixed_vs_gt = []
    fixed_vs_gt_mae = []
    alt_vs_gt_mae = []
    random_viewer_sample = np.random.randint(0,
                                             high=len(dataset),
                                             size=viewer_sample)
    for i in range(len(dataset)):
        # iterate over dataset
        gt, alt, flight_num, fl_path = dataset[i]

        # construct the altered flight if we haven't seen this flightpath already
        if curr_fl_path is not fl_path:
            curr_fl_path = fl_path
            print(f"Loading {fl_path}...")
            flight = File(fl_path)
            flight = np.stack([flight.x,
                               flight.y,
                               flight.z,
                               flight.intensity]).transpose(1, 0)

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
        alt = alt[1:, :] # chop out f1 point
        gt = gt[1:,:]
        query = kdtree._query(kd, alt.numpy(), k=tree_size)

        # index the altered flight to return clouds
        clouds = flight_a[query]  # tree_size, tree_size, 4

        # these clouds must be centered before inference
        clouds[:,:,:3] = np.stack([cloud[:,:3] - cloud[0][:3] for cloud in clouds])
        clouds = torch.tensor(clouds).cuda()

        # perform inference - get "fixed intensities"
        pred = torch.stack([model(cloud.unsqueeze(2))[0] for cloud in clouds])

        # After inference, can re-make alt with these new intensites. 
        fixed_sample = np.copy(alt)
        fixed_sample[:,3] = pred[0, :].cpu().detach().numpy()

        # Display fixed sample and alt side by side, only 3 times @ rand?
        if i in random_viewer_sample:
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

    # Create KDE Plot
    print("Generating KDE for visualization")
    code.interact(local=locals())
    

        

