from pathlib import Path

import numpy as np
import code
from multiprocessing import Pool
from functools import partial
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint

from src.datasets.tools.harmonization_mapping import HarmonizationMapping
from src.datasets.tools.metrics import create_kde
from src.datasets.tools.lidar_dataset import LidarDatasetNP
from src.datasets.tools.dataloaders import get_transforms
from src.config.pbar import get_pbar
from src.datasets.tools.transforms import GlobalShift, Corruption
from src.evaluation.histogram_matching import hist_match

from pptk import kdtree


# def harmonize(model, scan, harmonization_mapping, config):
def harmonize(model, source_scan_path, target_scan_num, config, save=False, sample_size=None):

    harmonized_path = Path(config['dataset']['harmonized_path'])
    plots_path = harmonized_path / "plots"
    plots_path.mkdir(exist_ok=True, parents=True)

    n_size = config['train']['neighborhood_size']
    b_size = config['train']['batch_size']
    chunk_size = config['dataset']['dataloader_size']
    transforms = get_transforms(config)
    G = GlobalShift(**config["dataset"])

    source_scan = np.load(source_scan_path)
    
    if config['dataset']['shift']:
            source_scan = G(source_scan)

    source_scan_num = int(source_scan[0, 8])
    
    if sample_size is not None:
        sample = np.random.choice(source_scan.shape[0], sample_size)
    else:
        sample = np.arange(source_scan.shape[0])

    model = model.to(config['train']['device'])
    model.eval()

    kd = kdtree._build(source_scan[:, :3])

    query = kdtree._query(
        kd, 
        source_scan[sample, :3], 
        k=n_size)

    query = np.array(query)
    size = len(query)
    
    hz = torch.empty(size).double()
    ip = torch.empty(size).double()
    cr = torch.empty(size).double()

    running_loss = 0

    pbar1 = get_pbar(
        range(0, len(query), chunk_size),
        int(np.ceil(source_scan.shape[0] / chunk_size)),
        f"Hzing Scan {source_scan_num}-->{target_scan_num}",
        0, leave=True, disable=config['dataset']['tqdm'])

    for i in pbar1:
        query_chunk = query[i:i+chunk_size, :]
        source_chunk = source_scan[i:i+chunk_size, :]
        source_chunk = np.expand_dims(source_chunk, 1)

        neighborhoods = np.concatenate(
            (source_chunk, source_scan[query_chunk]),
            axis=1)

        dataset = LidarDatasetNP(neighborhoods, transform=transforms)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=b_size,
            num_workers=config['train']['num_workers'])
        
        pbar2 = get_pbar(
            dataloader,
            len(dataloader),
            "  Processing Chunk",
            1, disable=config['dataset']['tqdm'])

        with torch.no_grad():
            for j, batch in enumerate(pbar2):
                batch[:, 0, -1] = target_scan_num  # specify that we wish to harmonize

                # batch = torch.tensor(np.expand_dims(ex, 0))
                batch = batch.to(config['train']['device'])

                # dublin specific?
                h_target = batch[:, 0, 3].clone()
                i_target = batch[:, 1, 3].clone()

                harmonization, interpolation, _ = model(batch)

                ldx = i + (j * b_size)
                hdx = i + (j+1)*b_size

                hz[ldx:hdx] = harmonization.cpu().squeeze()
                ip[ldx:hdx] = interpolation.cpu().squeeze()
                cr[ldx:hdx] = i_target.cpu() # corruption
  
                loss = torch.mean(torch.abs(harmonization.squeeze() - h_target))
                running_loss += loss.item()
                pbar2.set_postfix({"loss": f"{running_loss/(i+j+1):.3f}"})

    # visualize results
    hz = hz.numpy()
    hz = np.clip(hz, 0, 1)
    ip = ip.numpy()
    ip = np.clip(ip, 0, 1)
    cr = cr.numpy()
    cr = np.expand_dims(cr, 1)

    if config['dataset']['name'] == "dublin" and sample_size is None:
        create_kde(source_scan[sample, 3], hz.squeeze(),
                    xlabel="ground truth harmonization", ylabel="predicted harmonization",
                    output_path=plots_path / f"{source_scan_num}-{target_scan_num}_harmonization.png")

        create_kde(cr.squeeze(), ip.squeeze(),
                    xlabel="ground truth interpolation", ylabel="predicted interpolation",
                    output_path=plots_path / f"{source_scan_num}-{target_scan_num}_interpolation.png")

        create_kde(source_scan[sample, 3], cr.squeeze(),
                    xlabel="ground truth", ylabel="corruption",
                    output_path=plots_path / f"{source_scan_num}-{target_scan_num}_corruption.png")
    

    # insert results into original scan
    harmonized_scan = np.hstack((source_scan[sample, :3], np.expand_dims(hz, 1), source_scan[sample, 4:])) 

    if config['dataset']['name'] == "dublin":
        scan_error = np.mean(np.abs((source_scan[sample, 3]) - hz.squeeze()))
        print(f"Scan {source_scan_num} Harmonize MAE: {scan_error}")

    if save:
        np.save((Path(config['dataset']['harmonized_path']) / (str(source_scan_num)+".npy")), harmonized_scan)

    return harmonized_scan     

def harmonize_hm(source_scan_path, target_scan, config, save=False, sample_size=None):

    harmonized_path = Path(config['dataset']['harmonized_path'])
    plots_path = harmonized_path / "plots"
    plots_path.mkdir(exist_ok=True, parents=True)

    transforms = get_transforms(config)
    C = Corruption(**config['dataset'])
    G = GlobalShift(**config['dataset'])

    source_scan = np.load(source_scan_path)
    
    if config['dataset']['shift']:
            source_scan = G(source_scan)

    source_scan_num = int(source_scan[0, 8])
    
    if sample_size is not None:
        sample = np.random.choice(source_scan.shape[0], sample_size)
    else:
        sample = np.arange(source_scan.shape[0])

    size = len(source_scan.shape)
    
    ### perform harmonization here ###

    gt_intensity = source_scan[sample, 3].copy()
    alt_intensity = C(source_scan)[1:, 3][sample]  # chop off added point used in train
    fix_intensity = hist_match(alt_intensity, target_scan[:, 3])

    ### -------------------------- ###

    # visualize results
    if config['dataset']['name'] == "dublin":
        create_kde(gt_intensity, fix_intensity,
                    xlabel="ground truth harmonization", ylabel="predicted harmonization",
                    output_path=plots_path / f"{source_scan_num}-{config['dataset']['target_scan_num']}_harmonization.png")

        create_kde(gt_intensity, alt_intensity,
                    xlabel="ground truth", ylabel="corruption",
                    output_path=plots_path / f"{source_scan_num}-{config['dataset']['target_scan_num']}_corruption.png")
    

    # insert results into original scan
    harmonized_scan = np.hstack((source_scan[sample, :3], np.expand_dims(fix_intensity, 1), source_scan[sample, 4:])) 

    if config['dataset']['name'] == "dublin":
        scan_error = np.mean(np.abs(source_scan[sample, 3] - fix_intensity))
        print(f"Scan {source_scan_num} Harmonize MAE: {scan_error}")

    if save:
        np.save((Path(config['dataset']['harmonized_path']) / (str(source_scan_num)+".npy")), harmonized_scan)

    return harmonized_scan

if __name__ == "__main__":
    from src.datasets.dublin.config import config as dublin_config
    from src.training.config import config as train_config

    config = {
        "dataset": dublin_config,
        "train": train_config
    }

    hm = HarmonizationMapping(config)
    target_scan = np.load(config['dataset']['scans_path'] / (str(config['dataset']['target_scan']) + '.npy'))

    for source_scan_num in hm.get_stage(0):
        harmonized_scan = harmonize_hm(
                            hm[source_scan_num].source_scan_path.item(), 
                            target_scan,
                            config)

        np.save(str(hm.harmonization_path / (str(source_scan_num)+".npy")), harmonized_scan)
        hm.add_harmonized_scan_path(source_scan_num)
        hm.incr_stage(source_scan_num)
