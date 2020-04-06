import os
import time
import code
from pathlib import Path
import numpy as np
from pptk import kdtree
from multiprocessing import Pool

def load_laz(path):
    f = np.load(path)
    return f

def create_dataset(path,
                   neighborhood_size,
                   samples,
                   contains_flights=None,
                   output_prefix=None,
                   sanity_check=True):
    
    start_time = time.time()
    
    # get flight path files:
    laz_files_path = Path(path)
    laz_files = [file for file in laz_files_path.glob('*.npy')]
    file_count = len(laz_files)

    idx_count = 0
        
    # save path
    save_path = Path(r"%s_%s/gt/" % (neighborhood_size, samples))
    if output_prefix:
        save_path = Path(r"%s_%s_%s/gt/" % (output_prefix, neighborhood_size, samples))
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Created path: {save_path}")
    print(f"Found {file_count} flights")

    # load the first flight file
    f1 = np.load(laz_files[0])
    sample = np.random.choice(len(f1), size=samples)

    # Randomly sample the first flight
    f1_sample = f1[sample]
    
    p = Pool(1)

    if not contains_flights:
        flights_container = laz_files[1:]
    else:
        flights_container = [laz_files[i] for i in contains_flights]
                                    
    # Query flights 2-? with all points from flight 0
    for fidx, fi in enumerate(p.imap_unordered(load_laz, flights_container)):
        kd = kdtree._build(fi[:,:3])
        queries = kdtree._query(kd, f1_sample[:,:3], k=neighborhood_size, dmax=1)
        
        for idx, query in enumerate(queries):  # only get full neighborhoods
            if len(query) == neighborhood_size:
                fi_query = fi[query]

                if sanity_check:
                    check = np.linalg.norm(f1_sample[idx][:3] - fi_query[:,:3])
                    if check.any() > 1:
                        print("ERROR: points too far away")
                        
                # log the flight where this data comes from, the test point, and the query data
                sample = np.concatenate(
                    (np.expand_dims(f1_sample[idx], 0), fi_query)
                )
                
                np.save(save_path / f"{fidx}_{idx_count}.npy", sample)
                idx_count += 1
                

    print(f"Found {idx_count} total examples")
    print(f"finished in {time.time() - start_time} seconds")        

if __name__ == '__main__':
    import argparse
    
    create_dataset('dublin_flights',
                   200,
                   100000,
                   # contains_flights=[4],
                   # output_prefix="",
                   sanity_check=True)
    
