from pathlib import Path
from pptk import kdtree
from src.datasets.tools.dataloaders import get_transforms
import numpy as np
import code
from multiprocessing import Pool
from functools import partial
import torch
from tqdm import tqdm
from src.datasets.tools.lidar_dataset import LidarDatasetNP
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint


# def harmonize(model, scan, harmonization_mapping, config):
def harmonize(model, scan_path, target_scan, config):

    # scans = [f for f in Path(config['dataset']['scans_path']).glob("*.npy")]
    
    harmonized_path = Path(config['dataset']['harmonized_path'])
    plots_path = harmonized_path / "plots"
    plots_path.mkdir(exist_ok=True, parents=True)

    
    # scans_to_be_harmonized = {}
    # 
    #     # get the paths for each scan we wish to harmonize
    #     for source_scan_num in harmonization_mapping:
    #         for scan in scans:
    #             if int(scan.stem) == source_scan_num:
    #                 scans_to_be_harmonized[source_scan_num] = str(scan)


    #     for source_scan_num, scan_path in scans_to_be_harmonized.items():
    
    # I think this works to harmonize a single scan to a given target with model and config
    #   The above was trying to do too much. Move the above to the run script

    hz = []; ip = []; cr = []
    running_loss = 0
    n_size = model.neighborhood_size
    b_size = config['train']['batch_size']
    chunk_size = config['dataset']['max_chunk_size']
    transforms = get_transforms(config)
    transforms.transforms = transforms.transforms[1:] # no need to load np files

    source_scan = np.load(scan_path)

    model = model.to(config['train']['device'])
    model.eval()

    kd = kdtree._build(source_scan[:, :3])

    query = kdtree._query(
        kd, 
        source_scan[:, :3], 
        k=n_size)

    query = np.array(query)

    pbar1 = tqdm(
        range(0, len(query), chunk_size),
        desc="Processing Chunk",
        leave=False,
        position=0,
        dynamic_ncols=True,
        )

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

        pbar2 = tqdm(
            # range(len(dataset)),
            dataloader,
            desc="  Hzing dset",
            leave=False,
            position=1,
            dynamic_ncols=True)

        with torch.no_grad():
            for jdx, batch in enumerate(pbar2):
                # ex = dataset[j]

                # specify that we want to harmonize to the target scan:
                # ex[0, -1] = harmonization_mapping[int(ex[0, -1])]
                batch[:, 0, -1] = harmonization_mapping[int(batch[0, 0, -1])]

                # batch = torch.tensor(np.expand_dims(ex, 0))
                batch = batch.to(config['train']['device'])

                harmonization, interpolation, _, h_target, i_target = model(batch)
                hz.append(harmonization)   # interpolation
                ip.append(interpolation)   # harmonization
                cr.append(batch[:, 1, 3])  # corruption

                loss = torch.mean(torch.abs(harmonization.squeeze() - h_target))
                running_loss += loss.item()
                pbar2.set_postfix({"loss": f"{running_loss/(jdx+1):.3f}"})

    # visualize results
    hz = torch.cat(hz).cpu().numpy()
    ip = torch.cat(ip).cpu().numpy()
    cr = torch.cat(cr).cpu().numpy()

    create_kde(source_scan[:, 3], torch.cat(hp).cpu().numpy(),
                xlabel="ground truth", ylabel="predictions",
                output_path=plots_path)

    # insert results into original scan
    source_scan = np.hstack((source_scan[:, :4], hz, cr, ip, source_scan[:, 4:])) 
    np.save(harmonized_path / str(str(source_scan_num)+".npy"), source_scan)

            