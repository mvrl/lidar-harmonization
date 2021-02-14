# import order matters?
from src.datasets.dublin.tools.apply_rf import ApplyResponseFunction
from src.datasets.dublin.tools.metrics import create_kde

import code
import numpy as np
import pandas as pd
from pathlib import Path
from pptk import kdtree
from tqdm import tqdm, trange


def create_big_tile_dataset(path, neighborhood_size=150):

    path = Path(path)
    save_path = path / "neighborhoods"
    save_path.mkdir(parents=True, exist_ok=True) 
    ARF = ApplyResponseFunction("dorf.json", "mapping.npy")
    

    big_tile_gt = np.loadtxt(path / "gt.txt.gz")
    scan_num = int(big_tile_gt[0, 8])
    # big_tile_alt = np.load(path / "alt.npy")
    
    kd = kdtree._build(big_tile_gt[:, :3])

    query = kdtree._query(
            kd, 
            big_tile_gt[:, :3],
            k=neighborhood_size)

    my_query = []
    for i in query:
        if len(i) == neighborhood_size:
            my_query.append(i)

    good_sample_ratio = ((len(query) - len(my_query))/len(query)) * 100
    print(f"Found {good_sample_ratio} perecent of points with not enough close neighbors!")

    query = my_query
    examples = [None] * len(query)
    fid = [None] * len(query)
    intensities = [None] * len(query)

    # get neighborhoods
    for i in trange(len(query), desc="querying neighborhoods"):
        gt_neighborhood = big_tile_gt[query[i]]
        alt_neighborhood = ARF(gt_neighborhood, scan_num, 512)
        
        # Keep parity with training dataset - save example as (152, 9) point cloud
        # there will be an extra copy of the center pt altered gt at idx 1
        my_example = np.concatenate((
            np.expand_dims(gt_neighborhood[0, :], 0), 
            np.expand_dims(alt_neighborhood[0, :], 0),
            alt_neighborhood))
        
        np.savetxt(save_path / f"{i}.txt.gz", my_example)
        examples[i] = (save_path / f"{i}.txt.gz").absolute()
        fid[i] = my_example[0, 8]  # flight number
        intensities[i] = int(my_example[0, 4])

    # create csv
    df = pd.DataFrame()
    df["examples"] = examples
    df["source_scan"] = fid
    df["target_intensity"] = intensities

    df.to_csv(path / "big_tile_dataset.csv")


if __name__ == "__main__":
    print('starting...')
    # create_big_tile_dataset(r"big_tile_in_overlap")
    print("default:")
    create_big_tile_dataset(r"synth_crptn/big_tile_no_overlap")
    print("shift:")
    create_big_tile_dataset(r"synth_crptn+shift/big_tile_no_overlap")



