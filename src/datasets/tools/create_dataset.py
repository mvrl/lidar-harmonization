import code
import logging
import logging.config
import numpy as np
import pandas as pd
from torch import tensor
from pathlib import Path
from pptk import kdtree, viewer
from tqdm import tqdm, trange
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import partial


from src.datasets.tools.overlap import get_hist_overlap, get_overlap_points
from src.datasets.tools.overlap import neighborhoods_from_aoi, log_message, save_neighborhoods

def make_weights_for_balanced_classes(dataset, nclasses, config):
    # https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    count = [0] * nclasses
    for i in range(len(dataset)):
        label = dataset.iloc[i].target_intensity//config['igroup_size']
        count[label] += 1

    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])

    weight = [0] * len(dataset)
    for i in range(len(ataset)):
        label = dataset.iloc[i].target_intensity//config['igroup_size']
        weight[idx] = weight_per_class[label]

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

    val_split_point = len(df_test) - len(df_test)//2
    df_val = df_test.iloc[val_split_point:, :].reset_index(drop=True)
    df_test = df_test.iloc[:val_split_point, :].reset_index(drop=True)
    print(f"Training samples: {len(df_train)}")
    print(f"Validation samples: {len(df_val)}")
    print(f"Testing samples: {len(df_test)}")

    df_train.to_csv(dataset_path / "train.csv")
    df_val.to_csv(dataset_path / "val.csv")
    df_test.to_csv(dataset_path / "test.csv")

    return df_train, df_val, df_test


    print("Done.")


def create_eval_tile(config):

    scans_path = Path(config['scans_path'])
    save_path = Path(config['eval_save_path'])
    neighborhoods_path = save_path / "neighborhoods"
    neighborhoods_path.mkdir(parents=True, exist_ok=True)

    save_format = "{source_scan}_{target_scan}_{center}_{idx}.txt.gz"
    func = partial(save_neighborhood, neighborhoods_path, save_format)

    intersecting_flight = config['eval_source_scan']

    flight = np.load(scans_path / (intersecting_flight+".npy"))

    kd = kdtree._build(flight[:, :3])
    q = kdtree._query(kd, 
                      np.array([config['eval_tile_center']]),
                      k=config['eval_tile_size'])
    
    tile = flight[tuple(q)]

    q = kdtree._query(kd, tile[:, :3], k=config['max_n_size'])

    save_neighborhoods(tile, np.array(q), flight, func)

    examples = [f for f in neighborhoods_path.glob("*.txt.gz")]
    df = pd.DataFrame()
    df['examples'] = examples
    df.to_csv(save_path / "eval_dataset.csv")

    # v = viewer(tile[:, :3])
    # v.set(lookat=config['eval_tile_center'])


def create_dataset(targets, config):
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
    
    pbar_t = tqdm(targets, total=len(targets), dynamic_ncols=True)

    pbar_s = tqdm(
        scan_paths, 
        total=len(scan_paths.keys()), 
        position=1,
        leave=False,
        dynamic_ncols=True)
    pbar_s.set_description("Target | Source: [ | ]")

    original_target_scan_num = config['target_scan']
    scans_to_harmonize = []

    for target_scan_num in pbar_t:
        target_scan = np.load(scan_paths[target_scan_num])
        for source_scan_num in pbar_s:
            
            if ((source_scan_num in target_scans) or 
               (source_scan_num in scans_to_harmonize)):
                # don't process if the source is in `targets`. Additionally,
                #   don't process if it was already processed. This prevents 
                #   creating examples for the same source with multiple targets,
                #   which seems overly redundant.
                continue
            
            source_scan = np.load(scan_paths[source_scan_num])

            hist_info, _ = get_hist_overlap(target_scan, source_scan)

            # Overlap Region
            dsc = f"Target | Source: [{target_scan_num}|{source_scan_num}]"
            pbar.set_description(dsc)
            aoi = target_scan[get_overlap_points(target_scan, hist_info)]

            overlap_size = neighborhoods_from_aoi(
                aoi,
                source_scan,
                "ts",
                (target_scan_num,source_scan_num),
                func,
                scans_to_harmonize,
                logger=logger,
                **config)

            if overlap_size >= config['min_overlap_size']:
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

        # Create train-test splits, save as CSVS
        df_train, _, _ = make_csv(config)

        # Create training weights
        # `igroups is a relative number of classes. Regression is used in this 
        #    project, but a balance across the range of intensities is still 
        #    desired. 
        igroups = ceil(config['max_intensity']/config['igroup_size'])
        make_weights_for_balanced_classes(df_train, igroups)
        
        # if make_eval_tile:
        #     create_eval_tile(config)

