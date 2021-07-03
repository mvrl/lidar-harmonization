import numpy as np
from pathlib import Path

def get_physical_bounds(scans="dataset/dublin/npy", bounds_path="dataset/bounds.npy", reset=False):
    scans = Path(scans)
    if reset or not Path(bounds_path).exists():
        print("Generating new bounds")
        min_x = min_y = min_z = 99999999
        max_x = max_y = max_z = 1

        min_intensity = 0
        max_intensity = 512

        for pc in scans.glob("*.npy"):
            f = np.load(pc)
            if f[:, 0].min() < min_x:
                min_x = f[:, 0].min()
            if f[:, 0].max() > max_x:
                max_x = f[:, 0].max()
            if f[:, 1].max() < min_y:
                min_y = f[:, 1].min()
            if f[:, 1].max() > max_y:
                max_y = f[:, 1].max()
            if f[:, 2].min() < min_z:
                min_z = f[:, 2].min()
            if f[:, 2].max() > max_z:
                max_z = f[:, 2].max()

        bounds = np.array([[min_x, max_x], [min_y, max_y], [min_z, max_z]])
        print("Saved new bounds as ")
        print(bounds)
        np.save(str(bounds_path), bounds)
    else:
        bounds = np.load(str(bounds_path))

    return bounds

def sigmoid(x, h=0, v=0, s=1, l=1):
    return (s/(1 + np.exp(-l*(x-h)))) + v

def apply_shift_pc(pc, min_x, max_x):
    x = pc[:, 0]
    x = (x - min_x)/(max_x - min_x)

    floor = .3  # lower bound of sigmoid
    center = .5 # where the middle is
    pc[:, 3] = pc[:, 3] * sigmoid(x, h=center, v=floor, l=100, s=1-floor)
    return pc




