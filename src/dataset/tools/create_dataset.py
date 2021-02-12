import code
import logging
import logging.config
import numpy as np
import pandas as pd
from pathlib import Path
from pptk import kdtree
from tqdm import tqdm, trange
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import partial

from src.dataset.tools.overlap import get_hist_overlap, get_overlap_points
from src.dataset.tools.overlap import neighborhoods_from_aoi, log_message


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

def make_csv(config):
    print(f"building csv on {config['save_path']}")
    dataset_path = Path(config['save_path'])
    if dataset_path.exists():
        print(f"Found dataset folder at {config['save_path']}")
    
    examples = [f.absolute() for f in (dataset_path / "neighborhoods").glob("*.txt.gz")]
    if not len(examples):
        exit(f"could not find examples in {dataset_path / neighborhoods}")
    print(f"Found {len(examples)} examples")

    # Build a master record of all examples
    intensities = [None] * len(examples)
    source_scan = [None] * len(examples)
    target_scan = [None] * len(examples)
    
    for i in trange(len(examples), desc="processing"):
        filename = examples[i].stem
        source_scan[i] = filename.split("_")[0]
        target_scan[i] = filename.split("_")[1]
        intensities[i] = filename.split("_")[2]
    
    df = pd.DataFrame()
    df['examples'] = examples
    df['source_scan'] = source_scan
    df['target_scan'] = target_scan
    df['target_intensity'] = intensities

    cols = ['source_scan', 'target_scan', 'target_intensity']
    df[cols] = df[cols].apply(pd.to_numeric)
    print("Saving csv... ", end='')
    df.to_csv(dataset_path / "master.csv")
    print("Done.")
    
    # Create training/testing split
    print("Creating splits...")
    df = df.sample(frac=1).reset_index(drop=True)
    sample_count = len(df)
    split_point = sample_count - sample_count//5
    df_train = df.iloc[:split_point, :].reset_index(drop=True)
    df_test = df.iloc[split_point:, :].reset_index(drop=True)

    val_split_point = len(df_train) - len(df_train)//5
    df_val = df_train.iloc[val_split_point:, :].reset_index(drop=True)
    df_train = df_train.iloc[:val_split_point, :].reset_index(drop=True)
    print(f"Training samples: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")
    print(f"Testing samples: {len(df_test)}")
    print("Done.")

def create_dataset(config):
    # Neighborhoods path:   
    neighborhoods_path = Path(config['save_path']) / "neighborhoods"
    neighborhoods_path.mkdir(parents=True, exist_ok=True)

    # Logging
    logging.config.fileConfig('tools/logging.conf')
    logger = logging.getLogger('datasetCreation')
    logger.info("Starting")

    # Save neighborhoods as partial function
    save_format = "{source_scan}_{target_scan}_{center}_{idx}.txt.gz"
    func = partial(save_neighborhood, neighborhoods_path, save_format)

    # Get point clouds ready to load in
    scan_paths = {f.stem:f.absolute() for f in Path(config['scans_path']).glob("*.npy")}
    
    pbar = tqdm(scan_paths, total=len(scan_paths.keys()), dynamic_ncols=True)
    pbar.set_description("Total Progress [ | ]")

    # TO DO: target scan could be a collection of scans or tiles, so it will be
    #   necessary to check against a group of scans rather than just one. 
    #   Ideally, this variable would hold the largest segment of "target" 
    #   that is feasible. For dublin, it is simple to consider this as a single
    #   scan. However, to harmonize the entire region, harmonized scans will 
    #   become "target" scans. Futhermore, KY LiDAR and other tile-based LiDAR
    #   datasets will be another challenge in and of themselves.
    target_scan_num = config['target_scan']
    target_scan = np.load(scan_paths[target_scan_num])  

    for source_scan_num in pbar:
        if source_scan_num is target_scan_num:
            continue  # skip
        
        source_scan = np.load(scan_paths[source_scan_num])

        hist_info, _ = get_hist_overlap(target_scan, source_scan)

        # Overlap Region
        dsc = f"Total Progress [{target_scan_num}|{source_scan_num}]"
        pbar.set_description(dsc)
        aoi = target_scan[get_overlap_points(target_scan, hist_info)]

        overlap_size = neighborhoods_from_aoi(
            aoi,
            source_scan,
            "ts",
            (target_scan_num,source_scan_num),
            func,
            logger,
            **config)

        if overlap_size > config['min_overlap_size']:
            log_message("sufficent overlap, sampling outside", "INFO", logger)
            dsc = f"Total Progress [{source_scan_num}|{source_scan_num}]"
            pbar.set_description(dsc)
            aoi = source_scan[~get_overlap_points(source_scan, hist_info)]

            neighborhoods_from_aoi(
                aoi,
                source_scan,
                "ss",
                (target_scan_num,source_scan_num),
                func,
                logger,
                **config)

    # Wrap up
    make_csv(config)
