import os
import time
import code
from pathlib import Path
import numpy as np
from pptk import kdtree
from multiprocessing import Pool
from .patch import patch_mp_connection_bpo_17560

patch_mp_connection_bpo_17560()

def load_laz(path):
    f = np.load(path)
    return f

def create_dataset(path,
                   neighborhood_size,
                   samples,
                   example_count,
                   contains_flights=None,
                   output_suffix="",
                   sanity_check=True):
    
    start_time = time.time()
    
    # get flight path files:
    laz_files_path = Path(path)
    laz_files = [file for file in laz_files_path.glob('*.npy')]
    file_count = len(laz_files)

    idx_count = 0
        

    # save path
    save_path = Path(r"%s_%s%s/gt/" % (neighborhood_size, example_count, output_suffix))
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
        contains_flights = [i for i in range(41)]
    else:
        flights_container = [laz_files[i] for i in contains_flights]
    
    flight_counts = {i: 0 for i in range(len(flights_container))}
    # Query flights 2-? with all points from flight 0
    if neighborhood_size == 0:
        # single flight case, just copy the point
        for idx, sample in enumerate(f1_sample):
            if idx < example_count:
                # code.interact(local=locals())
                new_sample = np.expand_dims(sample, 0)
                new_sample = np.concatenate((new_sample, new_sample))
                np.save(save_path / f"1_{idx}.npy", new_sample)
            else:
                break  # early stop, found enough examples

    else:
        # dual and multi flight case, query for neighborhoods 
        for fidx, fi in enumerate(p.imap_unordered(load_laz, flights_container)):
            kd = kdtree._build(fi[:,:3])
            queries = kdtree._query(kd, f1_sample[:,:3], k=neighborhood_size, dmax=1)
        
            # code.interact(local=locals())   
            for idx, query in enumerate(queries):  # only get full neighborhoods
                if idx_count < example_count:
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
                
                        np.save(save_path / f"{contains_flights[fidx]}_{idx_count}.npy", sample)
                        idx_count += 1
                        flight_counts[fidx] += 1
                else:
                    break  # early stop, found enough examples
                
        print("Training example counts by flight:")
        print(flight_counts)
        print(f"Found {idx_count} total examples")
        if idx_count < example_count:
            print("Could not find enough examples! Try increasing sample size...")
            print("Only found {idx_count} samples")

    
    print(f"finished in {time.time() - start_time} seconds")        

if __name__ == '__main__':
    import argparse
    
    create_dataset('dublin_flights',
                   200,
                   100000,
                   # contains_flights=[4],
                   # output_prefix="",
                   sanity_check=True)
    
