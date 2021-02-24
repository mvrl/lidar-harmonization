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

def harmonize_neighborhoods(model, source_scan, query, batch_size=50, chunk_size=5000, pb_pos=1):
    # curr_idx = 1; max_idx = np.ceil(aoi.shape[0] / chunk_size)
    # sub_pbar = trange(0, aoi.shape[0], chunk_size,
 #                      desc=f"  Harmonizing", leave=False, position=pb_pos)

    # for i in sub_pbar:
 #      query_chunk = query[i:i+chunk_size, :]
    pass

        
def apply_harmonization(transform, data):
    return transform(data)


def harmonize(model, harmonization_mapping, config):

    scans = [f for f in Path(config['dataset']['scans_path']).glob("*.npy")]
    
    harmonized_path = Path(config['dataset']['harmonized_path'])
    harmonized_path.mkdir(exist_ok=True, parents=True)
    
    scans_to_be_harmonized = {}

    n_size = model.neighborhood_size
    b_size = config['train']['batch_size']
    transforms = get_transforms(config)
    transforms.transforms = transforms.transforms[1:] # no need to load np files
    t_func = partial(transforms)
    model.eval()


    # get the paths for each scan we wish to harmonize
    for source_scan_num in harmonization_mapping:
        for scan in scans:
            if int(scan.stem) == source_scan_num:
                scans_to_be_harmonized[source_scan_num] = str(scan)


    for source_scan_num, scan_path in scans_to_be_harmonized.items():
        source_scan = np.load(scan_path)

        kd = kdtree._build(source_scan[:, :3])

        query = kdtree._query(
            kd, 
            source_scan[:, :3], 
            k=n_size)

        query = np.array(query)
        code.interact(local=locals())
        pbar = tqdm(range(0, len(query), b_size), desc=f"{source_scan_num}")
        for i in pbar:
            batch_query = query[i:i+b_size, :] 
            batch = source_scan[batch_query]  # 50, N, 9
            center = batch[:, 0, :]
            center = np.expand_dims(center, 1)
            batch = np.concatenate((center, batch), axis=1)

            # apply transformation to batch.... 
            with Pool(config['train']['num_workers']) as p:
                batch = torch.stack([
                    torch.tensor(ex) for ex in p.imap_unordered(t_func, batch)
                    ])

            # specify that we want to harmonize to the target scan:
            for i in range(batch.shape[0]):
                batch[i, 0, -1] = harmonization_mapping[int(batch[i, 0, -1])]
            
            batch = batch.to(config['train']['device'])

            harmonization, interpolation, _, h_target, i_target = model(batch)

            loss = torch.mean(torch.abs(harmonization.squeeze() - h_target))
            pbar.set_postfix({"loss": f"{loss}"})





        