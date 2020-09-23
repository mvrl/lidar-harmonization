import os
import time
import code
from pathlib import Path
import numpy as np
from pptk import kdtree
from multiprocessing import Pool
import json
from src.dataset.tools.apply_rf import ApplyResponseFunction
import matplotlib.pyplot as plt
from tqdm import trange
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
                   dmax=0.5,
                   sanity_check=True):
    
    start_time = time.time()
    print(f"Starting dataset creation with dmax={dmax}")
    flight_counts = {}  

    # get flight path files:
    laz_files_path = Path(path)

    idx_count = 0
        
    # save path
    save_path = Path(r"%s_%s%s/" % (neighborhood_size, samples, output_suffix))
    save_path.mkdir(parents=True, exist_ok=True)

    example_path = save_path / "neighborhoods"
    example_path.mkdir(parents=True, exist_ok=True)
    print(f"Created path: {example_path}")
    
    mapping = load_mapping()
    APF = ApplyResponseFunction("dorf.json", "mapping.npy")
    
    # plot the response functions    
    # for f in contains_flights:
    #     x = np.linspace(0, 1, 1000)
    #     m = mapping[f]
    #     plt.plot(
    #             x, 
    #             np.interp(x, 
    #                 np.fromstring(rf_data[str(m)]['B'], sep=' '), 
    #                 np.fromstring(rf_data[str(m)]['I'], sep=' ')))
    # plt.plot(x, x, 'k')
    # plt.margins(x=0)
    # plt.margins(y=0)
    # plt.title("Response Functions in this dataset")
    # plt.savefig("response_plots.png")

    # load the target flight file
    f1 = np.load(laz_files_path / (str(base_flight)+".npy"))

    # Confirm we have flight 1 
    base_flight_num = f1[:, 8][0]
    target_flight = int(base_flight_num)
    print(f"Base flight is {base_flight_num} with {len(f1)} points")
    assert int(base_flight_num) == base_flight

    # Randomly sample the first flight
    # sample = np.random.choice(len(f1), size=samples)
    # f1_sample = f1[sample]

    # Just use the whole flight
    f1_sample = f1

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
        source_flight = int(fi[:, 8][0])

        print(f"Loaded flight #{source_flight}")

        flight_counts[source_flight] = 0
        kd = kdtree._build(fi[:,:3])

        # save memory by querying only 1/`num_incrememnts` at a time
        num_increments = 8
        start_idx = 0
        idx_increment = f1_sample.shape[0]//num_increments
        end_idx = start_idx + idx_increment
        for i in range(num_increments):
            queries = kdtree._query(
                    kd,
                    f1_sample[start_idx+(idx_increment*i):end_idx+(idx_increment*i), :3],
                    k=neighborhood_size,
                    dmax=.5)

            print(f"\tFound {len(queries)} queries") 

            for j in trange(len(queries)): # only get full neighborhoods
                query = queries[j]

                if len(query) == neighborhood_size:
                    fi_query = fi[query]
                    target_point = f1_sample[idx_increment*i + j]
                    assert np.linalg.norm(target_point[:3]-fi_query[0, :3]) <= dmax
                    target_intensity = int(target_point[3])
                    ground_truth = np.concatenate(
                            (np.expand_dims(target_point, 0), fi_query))
                    alteration = APF(ground_truth, source_flight, 512)
                
                    alteration = np.concatenate(
                        (np.expand_dims(target_point, 0), alteration)
                    )   
                
                    # Alteration will be of size (`neighborhood_size`+2, 9), with index=0 being
                    # the ground truth target center, and index=1 being the 
                    # altered copy of g.t. target center.

                    # code.interact(local=locals())
                    np.save(example_path / f"{source_flight}_{target_flight}_{str(target_intensity)}_{idx_count}.npy", alteration)

                    idx_count += 1
                    flight_counts[source_flight] += 1
            
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
    
    create_dataset('dublin/npy',
                   150,
                   35000000,
                   190001,
                   base_flight=1,
                   contains_flights=[0, 2, 4, 7, 10, 15, 20, 21, 30, 35, 37, 39],
                   # output_prefix="",
                   sanity_check=True,
                   dmax=.5)
