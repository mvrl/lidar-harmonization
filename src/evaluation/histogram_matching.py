import code
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from src.dataset.tools.metrics import create_kde
from src.dataset.tools.dataloaders import get_dataloader


def hist_match(target, reference):
    # # # 
    #
    # Params: 
    #    target: distribution as np array
    #    reference: distribution as np array
    #
    # Output: 
    #    "matched" target distribution to reference distribution
    #
    # # #
    
    # global mins and maxes
    g = np.concatenate((target, reference))

    # bins
    bin_range = [g.min(), g.max()]
    bin_num = int((g.max()-g.min())/5.)
    
    # Convert distributions to histograms
    target_hist, target_be = np.histogram(target, bins=bin_num, range=bin_range, density=True)
    reference_hist, reference_be = np.histogram(reference, bins=bin_num, range=bin_range, density=True)
    
    # PDF
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
    x = np.arange(g.min(), g.max(), (g.max()-g.min())/10000)

    m = np.interp(target, x, m_vals)
    
    return m


def hm_scans(tile, target_scan=1):

    tile_path = Path(tile)

    # load the tile
    tile_gt = np.load(tile_path)
    tile_alt = np.load(tile_path.parents[0] / "alt.npy")

    # Load the target scan
    target = np.load(f"dataset/dublin/npy/{int(target_scan)}.npy")
    
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
    print("MAE: ", MAE)

    np.savetxt(f"dataset/big_tile_no_overlap/fixed_hm.txt.gz", tile_fixed)

if __name__ == "__main__":
    hm_scans(
        "dataset/big_tile_no_overlap/gt.npy")

