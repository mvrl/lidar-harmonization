import code
import logging
import logging.config
import numpy as np
import pandas as pd
import torch
import time
from pathlib import Path
from pptk import kdtree, viewer
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import partial
import sharedmem
import h5py
from tqdm import tqdm, trange

from src.datasets.tools.transforms import GlobalShift
from src.datasets.tools.overlap import get_hist_overlap, get_overlap_points, get_pbar
from src.datasets.tools.overlap import neighborhoods_from_aoi, log_message, save_neighborhoods_hdf5, save_neighborhoods_hdf5_eval


def load_shared(path):
    t = np.load(path)
    ts = sharedmem.empty(t.shape)
    ts[:] = t
    return ts

def make_weights_for_balanced_classes(nclasses, config):
    # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    count = [0] * nclasses
    with h5py.File(config['dataset_path'], "a") as f:
        dataset = f['train']
        for i in range(dataset.shape[0]):
            label = int(np.floor(dataset[i][0, 3]/config['igroup_size']))  # hdf5
            count[label] += 1

        # code.interact(local=locals())
        weight_per_class = [0.] * nclasses
        N = float(sum(count))
        for i in range(nclasses):
            weight_per_class[i] = N/float(count[i])

        weight = [0] * len(dataset)
        for i in range(len(dataset)):
            label = int(np.floor(dataset[i][0, 3]/config['igroup_size']))  # hdf5
            weight[i] = weight_per_class[label]

    # note that weights do not have to sum to 1
    torch.save(weight, config['class_weights'])
    return weight

def save_neighborhood(path, save_format, data):
    idx, neighborhood = data

    target_scan_num = str(int(neighborhood[0, 8]))
    source_scan_num = str(int(neighborhood[1, 8]))
    center_intensity = str(int(neighborhood[0, 3]))

    save_string = save_format.format(
        source_scan=source_scan_num,
        target_scan=target_scan_num,
        center=center_intensity,
        idx=str(idx))

    np.savetxt(path / save_string, neighborhood)
    return

def setup_eval_hdf5(config):
    # just create the file
    if not config['eval_dataset'].exists():
        with h5py.File(config['eval_dataset'], "a") as f:
            eval_dset = f.create_dataset(
                "eval",
                (0, 151, 9),
                chunks=(config['max_chunk_size'], 151, 9),
                maxshape=(config['eval_tile_size'], 151, 9),
                dtype=np.float)


def create_eval_tile(config):
    
    scans_path = Path(config['scans_path'])
    intersecting_flight = config['eval_source_scan']
    flight = np.load(scans_path / (intersecting_flight+".npy"))
    
    G = GlobalShift(**config)

    if config['shift']:
        flight = G(flight)

    kd = kdtree._build(flight[:, :3])
    q = kdtree._query(kd, 
                      np.array([config['eval_tile_center']]),
                      k=config['eval_tile_size'])
    
    tile = flight[tuple(q)]

    q = kdtree._query(kd, tile[:, :3], k=config['max_n_size'])

    setup_eval_hdf5(config)

    save_neighborhoods_hdf5_eval(tile, 
                                np.array(q), 
                                flight, 
                                config,
                                chunk_size=500, # config['max_chunk_size']
                                pb_pos=0)


def setup_dataset_hdf5(config):

    # do nothing if we aren't recreating
    if config['create_new']:

        if config['dataset_path'].exists():
            print("Creating a new dataset! Deleteing the old one. ")
            print(config['dataset_path'])
            config['dataset_path'].unlink()  # delete
            config['create_new'] = False  # don't delete on another iteration

        # Size per scan is the igroup sample size times the number of intensity groups.
        #   Each scan is processed twice (out of overlap and in overlap), so x2
        #   The max size is the size per scan times the number of scans. 

        size_per_scan = 2*config['igroup_sample_size'] * int(np.ceil(1/config['igroup_size']))
        max_size = size_per_scan * len([f for f in Path(config['scans_path']).glob("*.npy")])
        print(size_per_scan)
        print(max_size)
        
        train_size = int(max_size * config['splits']['train'])
        test_size = int(max_size * config['splits']['test'])

        if not config['dataset_path'].exists():
            with h5py.File(config['dataset_path'], "a") as f:
                train_dset = f.create_dataset(
                    "train",
                    (0, 151, 9),
                    chunks=(config['max_chunk_size'], 151, 9),  # remove for auto-chunking
                    maxshape=(train_size, 151, 9),
                    dtype=np.float)

                test_dset = f.create_dataset(
                    "test",
                    (0, 151, 9),
                    chunks=(500, 151, 9),
                    maxshape=(test_size, 151, 9),
                    dtype=np.float)


def create_dataset(hm, config): 
    # Neighborhoods path:
    start=time.time()

    setup_dataset_hdf5(config)

    # Logging
    logging.config.fileConfig(config['creation_log_conf'])
    logger = logging.getLogger('datasetCreation')
    logger.info("Starting")

    # Shift
    G = GlobalShift(**config)

    # Get point clouds ready to load in
    h_scans_path = {f.stem:f.absolute() for f in Path(config['harmonized_path']).glob("*.npy")}
    scan_paths = {f.stem:f.absolute() for f in Path(config['scans_path']).glob("*.npy")}

    print("Creating dataset...")

    pbar_t = get_pbar(
        hm.get_stage(2),
        len(hm.get_stage(2)),
        "Building Dataset", 0, disable=config['tqdm'], leave=True)

    for target_scan_num in pbar_t:
        log_message(f"found target scan {target_scan_num}, checking for potential sources to harmonize", "INFO", logger)
        target_scan = load_shared(hm[target_scan_num].harmonization_scan_path.item())
        if config['shift']:
            target_scan = G(target_scan)
        
        pbar_s = get_pbar(
            hm.get_stage(0),
            len(hm.get_stage(0)),
            "Current: Target | Source: [ | ]", 1, 
            disable=config['tqdm'])

        for source_scan_num in pbar_s:
            log_message(f"found potential source scan {source_scan_num}, checking for overlap", "INFO", logger)
            source_scan = load_shared(hm[source_scan_num].source_scan_path.values[0])
            
            if config['shift']:
                source_scan = G(source_scan)

            hist_info, _ = get_hist_overlap(target_scan, source_scan)

            # Overlap Region
            pbar_s.set_description(f"Target | Source: [{target_scan_num}|{source_scan_num}]")
            aoi = target_scan[
                      get_overlap_points(
                          target_scan, 
                          hist_info, 
                          config, pb_pos=2)]

            overlap_size = neighborhoods_from_aoi(
                aoi,
                source_scan,
                "ts",
                (target_scan_num,source_scan_num),
                config,
                logger=logger)

            if overlap_size >= config['min_overlap_size']:
                # confirm that this scan can be harmonized to the current target
                hm.add_target(source_scan_num, target_scan_num)

                log_message(f"found sufficent overlap, sampling outside", "INFO", logger)
                pbar_s.set_description(f"Target | Source: [{source_scan_num}|{source_scan_num}]")
                aoi = source_scan[~get_overlap_points(source_scan, hist_info, config, pb_pos=2)]

                neighborhoods_from_aoi(
                    aoi,
                    source_scan,
                    "ss",
                    (target_scan_num,source_scan_num),
                    config,
                    logger)

                hm.incr_stage(source_scan_num)  # don't repeat this work
        hm.incr_stage(target_scan_num)
    

    igroups = int(np.ceil(config['max_intensity']/config['igroup_size']))
    print("Generating Class Weights")
    make_weights_for_balanced_classes(igroups, config)

    end=time.time()
    print(f"Created dataset in {end-start} seconds")
    # if make_eval_tile:
    #     create_eval_tile(config)

