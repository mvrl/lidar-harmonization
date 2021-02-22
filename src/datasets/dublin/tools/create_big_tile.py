import time
import code
import json
import numpy as np
from src.datasets.tools.metrics import create_kde

from pathlib import Path
from pptk import kdtree, viewer
import matplotlib.pyplot as plt
from src.config.project import Project
from src.datasets.dublin.tools.apply_rf import ApplyResponseFunction
from src.datasets.dublin.tools.shift import get_physical_bounds, apply_shift_pc
from src.datasets.dublin.config import config as dublin_config
from src.datasets.tools.transforms import Corruption, GlobalShift


def load_laz(path):
    f = np.load(path)
    return f

# TO DO: move this function
def visualize_n_flights(path, flights, sample_size=500000, shift=False):
    # flights is a list of numbers representing flights 0-40
    start_time = time.time()
    laz_files_path = Path(path)
    bounds = get_physical_bounds(scans=path, bounds_path="bounds.npy")
    
    pts = np.empty((0, 9))
    for scan in flights:
        
        fi = np.load(laz_files_path / (str(scan)+".npy"))
        if sample_size:
            sample = np.random.choice(len(fi), size=sample_size)
            pts = np.concatenate((pts, fi[sample])) 
        else:
            pts = np.concatenate((pts, fi))

    if shift:
        pts = apply_shift_pc(pts, bounds[0][0], bounds[0][1])
    attr1 = pts[:, 3]
    attr2 = pts[:, 8]

    v = viewer(pts[:, :3])
    v.attributes(attr1, attr2)
    # code.interact(local=locals())


