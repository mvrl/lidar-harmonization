import code
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from src.datasets.dublin.tools.shift import apply_shift_pc, sigmoid, get_physical_bounds

def hist_match(target, reference):
    # # # 
    #
    # Params: 
    #    target: distribution as np array (the distribution we want to fix)
    #    reference: distribution as np array 
    #
    # Output: 
    #    "matched" target distribution to reference distribution
    #
    # # #
    
    # global mins and maxes
    g = np.concatenate((target, reference))

    # bins can be a little tricky to choose. We use 512 since there are 512 
    # possible intensities (assuming intensity is a whole number)
    bin_range = [g.min(), g.max()]
    # bin_num = int((g.max()-g.min())/5.)
    bin_num = 512
    
    # Convert distributions to histograms
    target_hist, target_be = np.histogram(target, bins=bin_num, range=bin_range, density=True)
    reference_hist, reference_be = np.histogram(reference, bins=bin_num, range=bin_range, density=True)
    
    target_hist = target_hist/sum(target_hist)
    reference_hist = reference_hist/sum(reference_hist)
    
    # choose some arbitrary y values (y range: [0, 1])
    y_vals = np.random.uniform(size=10000)

    # sort these as monotonically increasing
    y_vals.sort()
    
    # interpolate x value pairs from the CDFs
    x_reference = np.interp(
        y_vals,
        np.hstack((np.zeros(1), np.cumsum(reference_hist))),
        reference_be)
    
    x_target = np.interp(
        y_vals,
        np.hstack((np.zeros(1), np.cumsum(target_hist))),
        target_be)
    
    # We now have three vectors denoting y-x0-x1 groups. We want to create a mapping
    # that defines the relationship for x0 -> x1 for any x0. 
    m_vals = np.interp(
        np.linspace(g.min(), g.max(), 10000),
        x_target,
        x_reference)
    
    # Interpolate values over the combined distributions
    x = np.linspace(g.min(), g.max(), 10000)

    m = np.interp(target, x, m_vals)
    
    return m


def hm_scans(tile, target_scan=1, shift=False):

    tile_path = Path(tile)

    file_type = tile_path.suffix

    tile_gt = np.loadtxt(tile_path)
    tile_alt = np.loadtxt(tile_path.parents[0] / "alt.txt.gz")

    # Load the target scan
    target = np.load(f"dataset/dublin/npy/{int(target_scan)}.npy")

    if shift:
        # apply global shift to the target scan
        bounds = get_physical_bounds()
        target = apply_shift_pc(target, bounds[0][0], bounds[0][1])

    # get distributions of intensities:
    d = [target[:, 3], tile_alt[:, 3]]

    # apply histogram matching 
    m = hist_match(d[1], d[0])

    # normalize the intensities for consistency with other methods
    tile_gt[:, 3] /= 512
    tile_alt[:, 3] /= 512
    target[:, 3] /= 512
    m /= 512

    # overwrite the target distribution with the hm-matched intensities
    tile_fixed = tile_alt.copy()
    tile_fixed = np.concatenate((
        tile_fixed[:, :3],             # XYZ
        tile_gt[:, 3].reshape(-1, 1),  # include ground truths
        m.reshape(-1, 1),              # fixed intensities
        tile_fixed[:, 4:]              # skip unaltered intensities
        ), axis=1)

    # calculate MAE
    MAE = np.mean(np.abs(m - tile_gt[:, 3]))
    save_path = tile_path.parents[0] / "fixed_hm.txt.gz"

    np.savetxt(str(save_path), tile_fixed)

    return MAE

if __name__ == "__main__":
    MAE = hm_scans("dataset/synth_crptn/big_tile_no_overlap/gt.txt.gz")
    print("default tile MAE:", MAE)
   
    MAE = hm_scans("dataset/synth_crptn+shift/big_tile_no_overlap/gt.txt.gz", shift=True)
    print("global shift MAE:", MAE)

