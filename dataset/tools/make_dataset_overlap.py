import os
import time
import code
from pathlib import Path
import numpy as np
from laspy.file import File
from pptk import kdtree
from multiprocessing import Pool

start_time = time.time()
    
# get flight path files:
laz_files_path = Path(r"/home/david/bin/python/dublin/dublaz/")
laz_files = [file for file in laz_files_path.glob('*')]
file_count = len(laz_files)

print(f"Found {file_count} flights")

f1_laz = File(laz_files[0])

f1 = np.stack([f1_laz.x, f1_laz.y, f1_laz.z, f1_laz.intensity]).T
sample = np.random.choice(len(f1), size=10000)
f1_sample = f1[sample]

def load_laz(path):
    f = File(path)
    fi = np.stack([f.x, f.y, f.z, f.intensity]).T
    f.close()
    return fi

p = Pool(3)

training_examples = []
num_examples = 0
for fidx, fi in enumerate(p.imap_unordered(load_laz, laz_files[1:])):

    kd = kdtree._build(fi[:,:3])
    queries = kdtree._query(kd, f1_sample[:,:3], k=50, dmax=1)
    for idx, query in enumerate(queries):
        if len(query) == 50:
            # log the flight where this data comes from, the test point, and the query data
            training_examples.append((fidx, f1_sample[idx], fi[query]))
    if (len(training_examples) > num_examples):
        print(f"Added {len(training_examples)-num_examples} examples")
        num_examples = len(training_examples)

    else:
        print("no overlap found!")

print(f"Found {len(training_examples)} total examples")

# sanity check these values:
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
        code.interact(local=locals())

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
        np.save(f"dataset/{key}_{idx}.npy", item)
    
print(f"finished in {time.time() - start_time} seconds")        

