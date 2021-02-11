import time
import code
import numpy as np
import pandas as pd
from pathlib import Path
from pptk import kdtree
from tqdm import tqdm, trange
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import partial
from src.dataset.tools.overlap import get_hist_overlap, get_overlap_points

from src.dataset.dublin.tools.apply_rf import ApplyResponseFunction
from src.dataset.dublin.tools.shift import get_physical_bounds, apply_shift_pc

def get_igroup_bounds(bin_size):
        return [(i, i+bin_size) for i in range(0, 512, bin_size)]


def save_neighborhood(path, save_format, data):
    idx, neighborhood = data
    idx = str(idx)
    target_scan_num = str(int(neighborhood[0, 8]))
    source_scan_num = str(int(neighborhood[1, 8]))
    center_intensity = str(int(neighborhood[0, 3]))

    save_string = save_format.format(
        source_scan=source_scan_num,
        target_scan=target_scan_num,
        center=center_intensity,
        idx=idx)

    np.savetxt(path / save_string, neighborhood)


if __name__ == "__main__":

    # Some options
    target_scan_num = '1'
    max_n_size = 150
    igroup_sample_size = 500  # sample this many points per "strata"
    igroup_size = 5

    # Setup
    igroup_bounds = get_igroup_bounds(igroup_size)
    max_chunk_size = int(4e6)

    # Dataset path:
    dataset_path = str(max_n_size)  # better name?
    dataset_path = Path(dataset_path)
    neighborhoods_path = dataset_path / "neighborhoods"
    neighborhoods_path.mkdir(parents=True, exist_ok=True)
    
    plots_path = dataset_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    save_format = "{source_scan}_{target_scan}_{center}_{idx}.txt.gz"

    # Save neighborhoods as partial function
    func = partial(save_neighborhood, neighborhoods_path, save_format)

    # Get point clouds ready to load in
    scans_dir = Path("npy/")
    scan_paths = {f.stem:f.absolute() for f in scans_dir.glob("*.npy")}

    # Load target (reference) scan
    target_scan = np.load(scan_paths[target_scan_num])

    pbar = tqdm(scan_paths, total=len(scan_paths.keys()), dynamic_ncols=True)
    pbar.set_description("Total Progress [ | ]")

    for source_scan_num in pbar:
        if source_scan_num is target_scan_num:
            continue  # skip
        
        # Load the next source scan        
        source_scan = np.load(scan_paths[source_scan_num])

        # build histogram - bins with lots of points in overlap likely exist
        #   in areas with a high concentration of other points in the overlap
        hist_info, _ = get_hist_overlap(target_scan, source_scan)

        for mode in ["ss", "ts"]:
            # obtain every point defined by the bin edges in `hist_info`
            if mode is "ts":
                # Overlap Region
                dsc = f"Total Progress [{target_scan_num}|{source_scan_num}]"
                pbar.set_description(dsc)
                aoi_idx = get_overlap_points(target_scan, hist_info)
                aoi = target_scan[aoi_idx]
            else:
                # Outside the overlap
                dsc = f"Total Progress [{source_scan_num}|{source_scan_num}]"
                pbar.set_description(dsc)
                aoi_idx = get_overlap_points(source_scan, hist_info)
                aoi = source_scan[~aoi_idx]

            # Get neighborhoods 
            kd = kdtree._build(source_scan[:, :3])

            # Querying uses a large amount of memory, use chunking to keep 
            #   the footprint small
            mfd = []; keep = []
            curr_idx = 0; max_idx = np.ceil(aoi.shape[0] / max_chunk_size)
            sub_pbar = trange(0, aoi.shape[0], max_chunk_size,
                              desc="  Filtering AOI", leave=False, position=1)
            for i in sub_pbar:
                current_chunk = aoi[i:i+max_chunk_size]
                query = kdtree._query(kd, 
                                      current_chunk[:, :3], 
                                      k=max_n_size, dmax=1)
    
                sub2_pbar = tqdm(range(len(query)),
                            desc=f"    Filtering [{curr_idx}/{max_idx}]",
                            leave=False,
                            position=2,
                            total=len(query))
            
                for j in sub2_pbar:
                    if len(query[j]) < max_n_size:
                        mfd.append(i+j)
                    else:
                        keep.append(i+j)

                curr_idx+=1

            # Select the points in target that have size `max_n_size`
            aoi_good = aoi[keep].copy()
            # save this information for future analysis
            bins = [i[0] for i in igroup_bounds] + [igroup_bounds[-1][1]]

            plt.hist(aoi_good[:, 3], bins)
            n = "Overlap" if "ts" else "Outside"
            plt.title("Pre-Sample Dist. of Intensities for src/trgt: "
                f"{source_scan_num}/{target_scan_num} - {n}.png")
            fname = f"{source_scan_num}_{target_scan_num}_{n}_post_i_dist.png"
            plt.savefig(str(plots_path / fname))
            plt.clf()

            # We want to resample the intensities here to be balanced
            #   across the range of intensities. 
            aoi_resampled = np.empty((0, aoi.shape[1]))
            sub_pbar = tqdm(igroup_bounds,
                            desc="  Resampling AOI",
                            leave=False,
                            position=1,
                            total=len(igroup_bounds))

            for (l, h) in sub_pbar:
                strata = aoi_good[(l <= aoi_good[:, 3]) & (aoi_good[:, 3] < h)]
                # auto append if strata is small
                if strata.shape[0] <= igroup_sample_size:
                    aoi_resampled = np.concatenate((
                        aoi_resampled, strata))

                # random sample if large
                else:
                    sample = np.random.choice(len(strata), igroup_sample_size)
                    aoi_resampled = np.concatenate((
                        aoi_resampled, strata[sample]))

            intensities = aoi_resampled[:, 3]
            plt.hist(intensities, bins)
            n = "Overlap" if "ts" else "Outside"
            plt.title("Post-Sample Dist. of Intensities for src/trgt: "
                f"{source_scan_num}/{target_scan_num} - {n}.png")
            fname = f"{source_scan_num}_{target_scan_num}_{n}_post_i_dist.png"
            plt.savefig(str(plots_path / fname))
            plt.clf()

            # build neighborhoods
            query = kdtree._query(kd, 
                                  aoi_resampled[:, :3], 
                                  k=max_n_size, dmax=1)

            query = np.array(query)

            # pair with target scan
            aoi_resampled = np.expand_dims(aoi_resampled, 1)
            neighborhoods = np.concatenate((aoi_resampled, 
                                            source_scan[query]), axis=1)

            pool = Pool(8)
            data = zip(range(neighborhoods.shape[0]), neighborhoods)
            sub_pbar = tqdm(pool.imap_unordered(func, data),
                        desc=f"Saving neighborhoods",
                        total=neighborhoods.shape[0],
                        position=1,
                        leave=False)
            for _ in sub_pbar:
                pass
            
            del neighborhoods 

