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


# TO DO: move this function
def visualize_n_flights(path, flights, sample_size=500000):
    start_time = time.time()
    laz_files_path = Path(path)

    G = GlobalShift(**dublin_config)
    C = Corruption(**dublin_config)

    pts = np.empty((0, 9))
    for scan in flights:
        fi = np.load(laz_files_path / (str(scan)+".npy"))
        fi = G(fi)  # apply global shift
        if sample_size:
            sample = np.random.choice(len(fi), size=sample_size)
            pts = np.concatenate((pts, fi[sample])) 
        else:
            pts = np.concatenate((pts, fi))

    attr1 = pts[:, 3]
    attr2 = pts[:, 8]

    v = viewer(pts[:, :3])
    v.attributes(attr1, attr2)
    # code.interact(local=locals())

if __name__ == "__main__":
    visualize_n_flights("data/test_npy", [1, 39])
