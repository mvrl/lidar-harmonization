import os
import time
import code
from pathlib import Path
import numpy as np
from pptk import kdtree
from multiprocessing import Pool
import json
from util.apply_rf import apply_rf
import matplotlib.pyplot as plt
from patch.patch import patch_mp_connection_bpo_17560

patch_mp_connection_bpo_17560()

def load_laz(path):
    f = np.load(path)
    return f

def random_mapping():
    mapping = np.random.choice(160, size=41)
    np.save("mapping.npy", mapping)
    return mapping

def load_mapping():
    mapping = np.load("mapping.npy")
    return mapping

def create_dataset(path,
                   neighborhood_size,
                   samples,
                   example_count,
                   base_flight=1,
                   contains_flights=None,
                   output_suffix="",
                   sanity_check=True):
    
    start_time = time.time()
    
    json_file = open("dorfCurves.json")
    rf_data = json.load(json_file)
    mapping = load_mapping() # random_mapping()
    flight_counts = {}  


    # get flight path files:
    laz_files_path = Path(path)

    idx_count = 0
        
    # save path
    save_path = Path(r"%s_%s%s/" % (neighborhood_size, example_count, output_suffix))
    save_path.mkdir(parents=True, exist_ok=True)
    gt_path = save_path / "gt"
    alt_path = save_path / "alt"
    gt_path.mkdir(parents=True, exist_ok=True)
    alt_path.mkdir(parents=True, exist_ok=True)

    print(f"Created path: {save_path}")

    # plot the response functions
    
    for f in contains_flights:
        x = np.linspace(0, 1, 1000)
        m = mapping[f]
        plt.plot(
                x, 
                np.interp(x, 
                    np.fromstring(rf_data[str(m)]['B'],sep=' '), 
                    np.fromstring(rf_data[str(m)]['I'], sep=' ')))
    plt.plot(x, x, 'k')
    plt.margins(x=0)
    plt.margins(y=0)
    plt.title("Response Functions in this dataset")
    plt.savefig("response_plots.png")



    # load the first flight file
    f1 = np.load(laz_files_path / (str(base_flight)+".npy"))

    # Confirm we have flight 1 
    base_flight_num = f1[:, 8][0]
    print(f"Base flight is {base_flight_num} with {len(f1)} points")
    assert int(base_flight_num) == base_flight


    # Randomly sample the first flight
    sample = np.random.choice(len(f1), size=samples)
    f1_sample = f1[sample]

    # contains flights will have to be defined
    flights_container = [laz_files_path / (str(i)+".npy") for i in contains_flights]
    
    laz_files_exist = [f.exists() for f in flights_container]
    if any(laz_files_exist) == False:
        exit(f"ERROR: files don't exist! Check {laz_files_path}")
    print(f"Found {len(flights_container)} flights")
    # Query flights with all points from flight 0
    # dual and multi flight case, query for neighborhoods 
    for fidx, flight_i in enumerate(flights_container):
        print(f"Loading flight file {flight_i}")
        fi = load_laz(flight_i)
        flight_num = int(fi[:, 8][0])
        print(f"Loaded flight #{flight_num}")

        flight_counts[flight_num] = 0
        
        kd = kdtree._build(fi[:,:3])
        queries = kdtree._query(kd,
                f1_sample[:,:3], 
                k=neighborhood_size, dmax=1/(2**1))

        for idx, query in enumerate(queries):  # only get full neighborhoods
            if len(query) == neighborhood_size:
                fi_query = fi[query]

                if sanity_check:
                    check = np.linalg.norm(f1_sample[idx][:3] - fi_query[:,:3])
                    if check.any() > 1:
                        print("ERROR: points too far away")
                
                sample = np.concatenate(
                    (np.expand_dims(f1_sample[idx], 0), fi_query)
                )   
        
                np.save(gt_path / f"{flight_num}_{idx_count}.npy", sample)
                altered_sample = apply_rf(
                        "dorfCurves.json", 
                        sample, 
                        mapping[flight_num],
                        512,
                        rf_data=rf_data)

                np.save(alt_path / f"{flight_num}_{idx_count}.npy", altered_sample)

                idx_count += 1
                flight_counts[flight_num] += 1
            
    print("Training example counts by flight:")
    print(flight_counts)
    print(f"Found {idx_count} total examples")
    if idx_count < example_count:
        print("Could not find enough examples! Try increasing sample size...")
        print(f"Only found {idx_count} samples")
    else:
        print("Found sufficient training data")
        print(mapping)
    
    print(f"finished in {time.time() - start_time} seconds")        

if __name__ == '__main__':
    import argparse
    
    create_dataset('dublin_flights/npy',
                   150,
                   20000000,
                   190000,
                   base_flight=1,
                   contains_flights=[0, 2, 4, 7, 10, 15, 20, 21, 30, 35, 37, 39],
                   # output_prefix="",
                   sanity_check=True)
    
