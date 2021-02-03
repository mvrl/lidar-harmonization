import time
import code
import numpy as np
import pandas as pd
from pathlib import Path
from pptk import kdtree
from tqdm import tqdm, trange
from src.dataset.tools.apply_rf import ApplyResponseFunction
from src.dataset.tools.shift import get_physical_bounds, apply_shift_pc


# Some options to play with
target_scan = '1'
overlap_sample_size = 10000
source_sample_size = 500000
overlap_threshold=150
workers=6
shift = True

# Setup
ARF = ApplyResponseFunction("dorf.json", "mapping.npy")
neighborhoods_path = Path("synth_crptn/150/neighborhoods")
neighborhoods_path.mkdir(parents=True, exist_ok=True)

if shift:
    shift_path = Path("synth_crptn+shift/150/neighborhoods")
    shift_path.mkdir(parents=True, exist_ok=True)

save_format = "{source_scan}_{target_scan}_{center}_{idx}.txt.gz"

def get_hist_overlap(pc1, pc2, s=10000, hist_bin_length=10):
    
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
    
    kd = kdtree._build(pc2[:, :3])

    sample_overlap = np.random.choice(len(pc1), size=s)
    pc1_sample = pc1[sample_overlap]

    query = kdtree._query(kd, pc1_sample[:, :3], k=150, dmax=1)
    
    counts = np.zeros((len(query), 1))
    for i in range(len(query)):
        counts[i][0] = len(query[i])

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
    
    indices = np.full(pc.shape[0], False, dtype=bool)
    hist, (xedges, yedges, zedges) = hist_info

    for i in trange(hist.shape[0], desc='building overlap region', leave=False, dynamic_ncols=True):
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


def balance_intensity(pc, example_count=2000):
    bins = np.arange(0, 520, 5)
    pc_final = np.empty((0, 9))
    for i in trange(len(bins)-1, desc='balancing overlap', leave=False, dynamic_ncols=True):
        left, right = bins[i], bins[i+1]
        pc_temp = pc.copy()
        pc_temp = pc_temp[(pc_temp[:, 3] < right) & (pc_temp[:, 3] >= left)]

        # sample if selection is large, otherwise just take it - oversample later?
        if pc_temp.shape[0] >= example_count:
            sample = np.random.choice(
                    len(pc_temp), 
                    size=example_count, 
                    replace=False)

            pc_temp = pc_temp[sample]
        
        pc_final = np.concatenate((pc_final, pc_temp))

    return pc_final


if __name__ == "__main__":

    # Get point clouds ready to load in 
    pc_dir = Path("dublin/npy/")
    pc_paths = {f.stem:f.absolute() for f in pc_dir.glob("*.npy")}
    
    pc1 = np.load(pc_paths[target_scan])

    # for each flight: 
    # 1. detect if there are overlaps
    # 2. if yes, find neighborhoods in the overlapping scan and 
    #    save them as examples with the center point from the target scan
    # 3. for each overlapping scan, also create examples from outside the 
    #    overlap region
    
    bounds = get_physical_bounds(scans="dublin/npy", bounds_path="bounds.npy")

    pbar = tqdm(pc_paths, total=len(pc_paths.keys()), dynamic_ncols=True)
    for source_scan in pbar:
        if source_scan is target_scan:
            continue  # skip
        
        pbar.set_description("Total Progress")

        # load pc2 
        pc2 = np.load(pc_paths[source_scan])
        
        # build histogram on overlap
        hist_info, _ = get_hist_overlap(pc1, pc2, s=overlap_sample_size)

        # collect all points in overlap bins w/ size > overlap_threshold
        overlap_indices = get_overlap_points(pc1, hist_info, overlap_threshold)

        pc_overlap = pc1[overlap_indices]

        if pc_overlap.shape[0] < 200000:
            continue  # overlap region too small, skip
    
        pc_overlap = balance_intensity(pc_overlap)
            
        # get neighborhoods from pc2 that contain 150 neighbors
        kd = kdtree._build(pc2[:, :3])
        query = kdtree._query(kd, pc_overlap[:, :3], k=150, dmax=1)

        # Store these in 150/neighborhoods/ with target intensity and
        # target intensity copy as the first two elements in the array.
        desc = "building neighborhoods from overlap"    
        for idx in trange(len(query), desc=desc, leave=False, dynamic_ncols=True):
            q = query[idx]

            if len(q) == 150:
                neighborhood = pc2[q]
 
                alt_neighborhood = ARF(neighborhood, int(source_scan), 512)
                
                center = np.expand_dims(pc_overlap[idx], 0)
                alt_center = ARF(center, int(source_scan), 512)
                ex = np.concatenate((center, alt_center, alt_neighborhood))

                save_string = save_format.format(
                    source_scan=source_scan,
                    target_scan=target_scan,
                    center=str(int(center[:,3])),
                    idx=idx)

                np.savetxt(neighborhoods_path / save_string, ex)
                
                if shift:
                    # apply the shift to the neighborhood *before* the corruption
                    shift_neighborhood = apply_shift_pc(
                        neighborhood.copy(), 
                        bounds[0][0], bounds[0][1])

                    alt_shift_neighborhood = ARF(neighborhood, int(source_scan), 512)
                    shift_center = apply_shift_pc(center, bounds[0][0], bounds[0][1])
                    alt_shift_center = ARF(shift_center, int(source_scan), 512)
                    shift_ex = np.concatenate((shift_center, 
                                               alt_shift_center, 
                                               alt_shift_neighborhood))

                    save_string = save_format.format(
                        source_scan=source_scan,
                        target_scan=target_scan,
                        center=str(int(shift_center[:,3])),
                        idx=idx)

                    np.savetxt(shift_path / save_string, shift_ex)
         
        # sample size between source & overlap might be an issue. Fix in post?
        source_sample_idx = np.random.choice(pc2.shape[0], size=source_sample_size)
        pc_source = balance_intensity(pc2[source_sample_idx])

        query = kdtree._query(kd, pc_source[:, :3], k=150, dmax=1)
        desc="building neighborhoods from source"
        for idx in trange(len(query), desc=desc, leave=False, dynamic_ncols=True):
            q = query[idx]

            if len(q) == 150:
                neighborhood = pc2[q]
                center = np.expand_dims(neighborhood[0], 0)
                alt_neighborhood = ARF(neighborhood, int(source_scan), 512)
                alt_center = np.expand_dims(alt_neighborhood[0], 0)
                ex = np.concatenate((center, alt_center, alt_neighborhood))

                save_string = save_format.format(
                    source_scan=source_scan,
                    target_scan=source_scan,
                    center=str(int(center[:,3])),
                    idx=idx)

                np.savetxt(neighborhoods_path / save_string, ex)

                if shift: 
                    shift_neighborhood = apply_shift_pc(
                        neighborhood.copy(), 
                        bounds[0][0], bounds[0][1])

                    alt_shift_neighborhood = ARF(neighborhood, int(source_scan), 512)
                    shift_center = apply_shift_pc(center, bounds[0][0], bounds[0][1])
                    alt_shift_center = ARF(shift_center, int(source_scan), 512)
                    shift_ex = np.concatenate((shift_center, 
                                               alt_shift_center, 
                                               alt_shift_neighborhood))
                    
                    save_string = save_format.format(
                        source_scan=source_scan,
                        target_scan=source_scan,
                        center=str(int(shift_center[:,3])),
                        idx=idx)

                    np.savetxt(shift_path / save_string, shift_ex)



