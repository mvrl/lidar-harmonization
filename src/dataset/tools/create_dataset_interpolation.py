import time
from pptk import kdtree, viewer
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm, trange
from src.dataset.tools.apply_rf import ApplyResponseFunction
import code

def create_dataset(path, mapping, contains_flights=None):
    # Create dataset from all scans for interpolation learning
    start_time = time.time()
    
    N_SIZE = 151  # 152 = N_SIZE + 1, N_SIZE = 151
    scans_base_path = Path(path) / "npy"
    scans_paths = [scan_path for scan_path in scans_base_path.glob("*.npy")]
    
    save_path = Path("interpolation_dataset").absolute()
    save_path.mkdir(parents=True, exist_ok=True)
    save_path_ns = save_path / "neighborhoods"
    save_path_ns.mkdir(parents=True, exist_ok=True)

    if contains_flights is None:
        contains_flights = np.arange(42)

    APF = ApplyResponseFunction("dorf.json", mapping)
    count = 0
    for i in range(len(scans_paths)):
        if int(scans_paths[i].stem) in contains_flights:
            print(f"Loading scan {scans_paths[i]}...",end='')
            scan = np.load(scans_paths[i])
            fid = int(scan[0, 8])
            print(f"Loaded scan {fid}")

            scan = APF(scan, fid, 512)
            sample = np.random.choice(len(scan), size=int(10e5), replace=False)
            scan_sample = scan[sample]
            
            kd = kdtree._build(scan[:, :3])
            queries = kdtree._query(kd, scan_sample[:, :3], k=N_SIZE, dmax=1)
            
            for i in trange(len(queries)):
               query = queries[i]
               if len(query) == N_SIZE:

                   # only take full neighborhoods
                           
                   # the target point exists in the neighborhood 
                   # but will be removed at train/eval time
                   target_point = scan_sample[i]
                   target_i = int(target_point[3])
                   
                   # append target point to neighborhood
                   # note that the first two points are the same,
                   # and will have the same camera. This should 
                   # inform training that no harmonization needs to occur.
                   example = np.concatenate((
                       np.expand_dims(target_point, 0),
                       scan[query]))

                   np.save(
                           save_path_ns / f"{fid}_{str(target_i)}_{i}.npy", 
                           example
                          )
                   count += 1


    print(f"Generated {count} samples in {time.time() - start_time} seconds")
    


if __name__ == "__main__":
    create_dataset(
            "dublin", 
            "mapping.npy", 
            # contains_flights=[0, 1, 2, 4, 7, 10, 15, 20, 21, 30, 35, 37, 39],
            contains_flights=[1])



