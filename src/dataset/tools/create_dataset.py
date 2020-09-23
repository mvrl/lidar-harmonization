import time
import code
import numpy as np
import pandas as pd
from pathlib import Path
from pptk import kdtree
from tqdm import tqdm, trange
from src.dataset.tools.apply_rf import ApplyResponseFunction


def get_hist_overlap(pc1, pc2, sample_overlap_size=10000, hist_bin_length=10):
    # Params:
    #     pc1: point cloud 1 (np array with shape ([m, k1]))
    #     pc2: point cloud 2 (np array with shape ([n, k2]))
    #
    # k1 and k2 must contain at least x and y coordinates. 
    
    #
    # Returns:
    #     
    
    # define a data range
    pc_combined = np.concatenate((pc1, pc2))
    data_range = np.array(
        [[pc_combined[:, 0].min(), pc_combined[:, 0].max()],
        [pc_combined[:, 1].min(), pc_combined[:, 1].max()],
        [pc_combined[:, 2].min(), pc_combined[:, 2].max()]])

    del pc_combined  # save some mem
    
    # define bins based on data_range:
    x_bins = int((data_range[0][1]-data_range[0][0])/10)
    y_bins = int((data_range[1][1]-data_range[1][0])/10)
    z_bins = int((data_range[2][1]-data_range[2][0])/10)
    
    # Collect some number of points as overlap between these point clouds
    # build kd tree so we can search for points in pc2
    kd = kdtree._build(pc2[:, :3])

    # collect a sample of points in pc1 to query in pc2
    sample_overlap = np.random.choice(len(pc1), size=sample_overlap_size)
    pc1_sample = pc1[sample_overlap]

    # query pc1 sample in pc2. note that we want lots of nearby neighbors
    query = kdtree._query(kd, pc1_sample[:, :3], k=150, dmax=1)
    
    # Count the number of neighbors found at each query point
    counts = np.zeros((len(query), 1))
    for i in range(len(query)):
        counts[i][0] = len(query[i])

    # Append this to our sample
    pc1_sample_with_counts = np.concatenate((pc1_sample[:, :3], counts), axis=1)

    # this needs to be transformed such that the points (X, Y) occur in the
    # array `count` times. This will make histogram creation easier.
    rows = []
    for i in range(len(pc1_sample_with_counts)):
        row = pc1_sample_with_counts[i, :3]
        row = np.expand_dims(row, 0)
        if pc1_sample_with_counts[i, 2]:
            duplication = np.repeat(row, pc1_sample_with_counts[i, 3], axis=0)
            rows.append(duplication)
    
    pc1_sample_f = np.concatenate(rows, axis=0)
    
    # build histogram over data
    hist, edges = np.histogramdd(
        pc1_sample_f[:, :3], 
        bins=[x_bins, y_bins, z_bins],
        range=data_range)

    return (hist, edges), pc1_sample_f


def get_overlap_points(pc, hist_info, c):
    # this seems slow
    
    indices = np.full(pc.shape[0], False)
    hist, (xedges, yedges, zedges) = hist_info

    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            for k in range(hist.shape[2]):
                if hist[i][j][k] > c:
                    x1, x2 = xedges[i], xedges[i+1]
                    y1, y2 = yedges[j], yedges[j+1]
                    z1, z2 = zedges[k], zedges[k+1]
                    
                    new_indices = ((x1 <= pc[:, 0]) & (pc[:, 0] < x2) & 
                        (y1 <= pc[:, 1]) & (pc[:, 1] < y2) &
                        (z1 <= pc[:, 2]) & (pc[:, 2] < z2))
                    
                    indices = indices | new_indices

    return indices


if __name__ == "__main__":

    # Some options to play with
    sample_overlap_size = int(10e3)
    sample_not_overlap_size = int(3e3)

    # Setup
    ARF = ApplyResponseFunction("dorf.json", "mapping.npy")
    neighborhoods_path = Path("150/neighborhoods")
    neighborhoods_path.mkdir(parents=True, exist_ok=True)

    # Get point clouds ready to load in 
    pc_dir = Path("dublin/npy/")
    pc_paths = {f.stem:f.absolute() for f in pc_dir.glob("*.npy")}

    # choose a base flight as "target flight"
    target_scan = '1'
    pc1 = np.load(pc_paths[target_scan])

    # for each flight: 
    # 1. detect if there are overlaps
    # 2. if yes, find neighborhoods in the overlapping scan and 
    #    save them as examples with the center point from the target scan
    # 3. for each overlapping scan, also create examples from outside the 
    #    overlap region
    # 4. necessary to have examples from target flight?
    for scan_num in pc_paths:
        if scan_num is not target_scan:

            # load pc2
            pc2 = np.load(pc_paths[scan_num])
            
            # build histogram on overlap
            hist_info, _ = get_hist_overlap(pc1, pc2)

            # collect all points in overlap bins w/ size > 150
            overlap_indices = get_overlap_points(pc1, hist_info, 150)
            
            pc_overlap = pc1[overlap_indices]
            pc_not_overlap = pc1[~overlap_indices]

            # don't do anyting if no overlap region is found
            if pc_overlap.shape[0] == 0:
                continue
    
            # get neighborhoods from pc2 that contain 150 neighbors
            kd = kdtree._build(pc2[:, :3])
            query = kdtree._query(kd, pc_overlap[:, :3], k=150, dmax=1)

            # any query with len(query) == 150 is a full neighborhood.
            # Store these in 150/neighborhoods/ with target intensity and
            # target intensity copy as the first two elements in the array.

            for idx, q in enumerate(query):
                if len(q) == 150:
                    neighborhood = pc2[q]
                    alt_neighborhood = ARF(neighborhood, int(scan_num), 512)
                    center = np.expand_dims(pc_overlap[idx], 0)
                    alt_center = ARF(center, int(scan_num), 512)

                    ex = np.concatenate((
                        center, 
                        alt_center, 
                        alt_neighborhood))

                    # save format is 
                    # neighborhoods/{source}_{target}_{target_intensity}
                    np.save(neighborhoods_path / 
                            f"{scan_num}_{target_scan}_{int(alt_center[:, 3])}_{idx}.npy",
                            ex)
            
            # one potential issue from this is that we will inevitably get
            # a much larger sample from this vs the samples we acquired in the
            # steps above. Maybe just use a constant value.

            sample_not_overlap_idx = np.random.choice(
                    len(pc2), 
                    size=sample_not_overlap_size, 
                    replace=False)
            
            sample_not_overlap = pc2[sample_not_overlap_idx]

            # note that k=151 so we have consistent size with overlap samples
            query = kdtree._query(kd, sample_not_overlap[:, :3], k=151, dmax=1)
            for idx, q in enumerate(query):
                if len(q) == 151:  # not sure why this wouldn't be the case
                    neighborhood = pc2[q]
                    center = np.expand_dims(neighborhood[0], 0)
                    alt_neighborhood = ARF(neighborhood, int(scan_num), 512)

                    ex = np.concatenate((center, alt_neighborhood))
                    np.save(neighborhoods_path / 
                            f"{scan_num}_{scan_num}_{int(neighborhood[0, 4])}.npy",
                            ex)

                    
                    



        








    

