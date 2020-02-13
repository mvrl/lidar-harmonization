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


def create_dataset(path, neighborhood_size, samples, sanity_check=True):
    start_time = time.time()
    
    # get flight path files:
    laz_files_path = Path(path)
    laz_files = [file for file in laz_files_path.glob('*.npy')]
    file_count = len(laz_files)

    # save path
    save_path = Path(r"%s_%s/gt/" % (neighborhood_size, samples))
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Created path: {save_path}")
    print(f"Found {file_count} flights")

    # load the first flight file
    f1 = np.load(laz_files[0])
    sample = np.random.choice(len(f1), size=samples)

    # Randomly sample the first flight
    f1_sample = f1[sample]
    
    p = Pool(4)

    training_examples = []
    num_examples = 0

    # Query flights 2-? with all points from flight 1
    for fidx, fi in enumerate(p.imap_unordered(load_laz, laz_files[1:])):
        kd = kdtree._build(fi[:,:3])
        queries = kdtree._query(kd, f1_sample[:,:3], k=neighborhood_size, dmax=1)
        for idx, query in enumerate(queries):
            if len(query) == neighborhood_size:
                # log the flight where this data comes from, the test point, and the query data
                training_examples.append((fidx, f1_sample[idx], fi[query]))
        if (len(training_examples) > num_examples):
            print(f"Added {len(training_examples)-num_examples} examples")
            num_examples = len(training_examples)

        else:
            print("no overlap found!")

    print(f"Found {len(training_examples)} total examples")

    # sanity check these values:
    if sanity_check:
        print("running sanity check...")
        flag = 0
        for idx, ex in enumerate(training_examples):
            a = ex[1][:3]
            b = ex[2][:,:3]
            c = np.linalg.norm(b - a, axis=1)
            # Distance from a should never be greater than dmax=1
            if c.any() > 1:
                print("ERROR")
                flag = 1
        if flag == 0:
            print("no issues detected")

    # save this as something useable for later
    # group these by flight
    dataset = {}

    for i in range(len(training_examples)):
        name = training_examples[i][0]
        a = np.expand_dims(training_examples[i][1], 0)
        b = training_examples[i][2]
        if name in dataset:
            dataset[name].append(np.concatenate((a, b)))
        else:
            dataset[name] = [np.concatenate((a, b))]

    for key, datalist in dataset.items():
        for idx, item in enumerate(datalist):
            np.save(save_path / f"{key}_{idx}.npy", item)
    
    print(f"finished in {time.time() - start_time} seconds")        

if __name__ == '__main__':
    import argparse
    
    create_dataset('dublin_flights', 50, 10000, sanity_check=True)
