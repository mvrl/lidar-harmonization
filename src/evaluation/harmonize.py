from pathlib import Path
from pptk import kdtree
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

from src.datasets.tools.metrics import create_kde
from src.datasets.tools.lidar_dataset import LidarDatasetNP
from src.datasets.tools.dataloaders import get_transforms
from src.config.pbar import get_pbar


# def harmonize(model, scan, harmonization_mapping, config):
def harmonize(model, source_scan_path, target_scan_num, config, sample_size=None):

    harmonized_path = Path(config['dataset']['harmonized_path'])
    plots_path = harmonized_path / "plots"
    plots_path.mkdir(exist_ok=True, parents=True)

    hz = torch.empty(0).double() ; ip = torch.empty(0).double() ; cr = torch.empty(0).double() 
    running_loss = 0
    n_size = model.neighborhood_size
    b_size = config['train']['batch_size']
    chunk_size = config['dataset']['max_chunk_size']
    transforms = get_transforms(config)
    transforms.transforms = transforms.transforms[1:] # remove LoadNP step

    source_scan = np.load(source_scan_path)
    source_scan_num = int(source_scan[0, 8])
    
    # This might be in the wrong spot
    # if sample_size is not None:
    #     sample = np.random.choice(source_scan.shape[0], sample_size)
    #     source_scan = source_scan[sample]

    model = model.to(config['train']['device'])
    model.eval()

    kd = kdtree._build(source_scan[:, :3])

    query = kdtree._query(
        kd, 
        source_scan[:, :3], 
        k=n_size)

    query = np.array(query)
    
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
            "Processing Chunk",
            1, disable=config['dataset']['tqdm'])

        with torch.no_grad():
            for jdx, batch in enumerate(pbar2):
                batch[:, 0, -1] = target_scan_num  # specify that we wish to harmonize

                # batch = torch.tensor(np.expand_dims(ex, 0))
                batch = batch.to(config['train']['device'])

                # dublin specific?
                h_target = batch[:, 0, 3].clone()
                i_target = batch[:, 1, 3].clone()

                harmonization, interpolation, _ = model(batch)

                hz = torch.cat((hz, harmonization.cpu()))  # interpolation
                ip = torch.cat((ip, interpolation.cpu()))  # harmonization
                cr = torch.cat((cr, i_target.cpu()))       # corruption

                loss = torch.mean(torch.abs(harmonization.squeeze() - h_target))
                running_loss += loss.item()
                pbar2.set_postfix({"loss": f"{running_loss/(jdx+1):.3f}"})

    # visualize results
    hz = hz.numpy()
    hz = np.clip(hz, 0, 1)
    ip = ip.numpy()
    ip = np.clip(ip, 0, 1)
    cr = cr.numpy()
    cr = np.expand_dims(cr, 1)

    if config['dataset']['name'] == "dublin":
        create_kde(source_scan[:, 3]/512, hz.squeeze(),
                    xlabel="ground truth harmonization", ylabel="predicted harmonization",
                    output_path=plots_path / f"{source_scan_num}-{target_scan_num}_harmonization.png")

        create_kde(cr.squeeze(), ip.squeeze(),
                    xlabel="ground truth interpolation", ylabel="predicted interpolation",
                    output_path=plots_path / f"{source_scan_num}-{target_scan_num}_interpolation.png")

        create_kde(source_scan[:, 3]/512, cr.squeeze(),
                    xlabel="ground truth", ylabel="corruption",
                    output_path=plots_path / f"{source_scan_num}-{target_scan_num}_corruption.png")
    

    # insert results into original scan
    harmonized_scan = np.hstack((source_scan[:, :3], hz, source_scan[:, 4:])) 
    return harmonized_scan     
