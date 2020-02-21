import time
import code
import json
import numpy as np
from pathlib import Path
from pptk import kdtree

from util.apply_rf import apply_rf
from create_dataset import load_laz

def create_big_tile(path, alteration, size):

    start_time = time.time()

    # get the laz files:
    laz_files_path = Path(path)
    laz_files = [f for f in laz_files_path.glob('*.npy')]
    file_count = len(laz_files)

    # save path
    save_path = Path(r"big_tile/")
    save_path.mkdir(exist_ok=True)

    # load the first flight
    f1 = np.load(laz_files[0])

    # get a random point
    sample = np.random.choice(len(f1), size=1)
    f1_sample = f1[sample]

    # Build a giant kd tree
    kd = kdtree._build(f1[:, :3])
    query = kdtree._query(kd, f1_sample[:, :3], k=size)

    big_tile = f1[query]

    np.save(save_path / "big_tile.npy", big_tile)

    # now mess up the big tile

    big_tile_alt = apply_rf("response_functions.json", big_tile, alteration)

    np.save(save_path / "big_tile_alt.npy", big_tile_alt)

    print(f"finished in {time.time() - start_time} seconds")
    

if __name__=='__main__':
    create_big_tile('dublin_flights', 4, 1000000)

    
                 
    
