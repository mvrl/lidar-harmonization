import os
import json
import time
import code
from pathlib import Path
import numpy as np
from pptk import kdtree
from pptk import rand


# Create a synthetic dataset to verify matching
flight_num = 2
psize = 100000
intsize = 512
sample_size = 10000
neighborhood_size = 20


# Create synthetic "ground truth" data
xyz = rand(psize, 3)

# define some intensity values
gt_int = np.random.randint(0, high=512, size=psize)

gt_int = np.expand_dims(gt_int, axis=1)

gt = np.concatenate((xyz, gt_int), axis=1)

sample_indices = np.random.choice(np.arange(psize), sample_size)

# for each sample, create a neighborhood 
samples = []
kd = kdtree._build(gt[:,:3])
query = kdtree._query(kd, sample_indices, neighborhood_size, dmax=1)

# get gt and alt versions
gt_samples = gt[query]

for idx, i in enumerate(gt_samples):
    np.save(f"synth/gt/2_{idx}.npy", i)


