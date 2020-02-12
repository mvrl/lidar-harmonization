import time
import code
import json
import numpy as np
from pathlib import Path
from pptk import kdtree

from create_dataset import load_laz

def create_big_tile(path):

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
    query = kdtree._query(kd, f1_sample[:, :3], k=30000)

    big_tile = f1[query]

    np.save(save_path / "big_tile.npy", big_tile)

    # now mess up the big tile
    with open('response_functions.json') as json_file:
        data = json.load(json_file)

    big_tile_intensities = big_tile[:,3].copy()
    alt_big_tile_intensities = np.interp(big_tile_intensities,
                                         np.array(data['brightness']['2'])*255,
                                         np.array(data['intensities']['2'])*255)

    alt_big_tile = big_tile.copy()
    alt_big_tile[:, 3] = alt_big_tile_intensities

    np.save(save_path / "big_tile_alt.npy", alt_big_tile)

    print(f"finished in {time.time() - start_time} seconds")
    

if __name__=='__main__':
    create_big_tile('dublin_flights')

    
                 
    
