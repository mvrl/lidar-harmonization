import time
import code
import numpy as np
import pandas as pd
from pathlib import Path
from pptk import kdtree
from tqdm import tqdm, trange
import matplotlib.pyplot as plt

from src.dataset.tools.overlap import get_hist_overlap, get_overlap_points

from src.dataset.dublin.tools.apply_rf import ApplyResponseFunction
from src.dataset.dublin.tools.shift import get_physical_bounds, apply_shift_pc

def get_igroup_bounds(bin_size):
        return [(i, i+bin_size) for i in range(0, 512, bin_size)]


def build_neighborhoods(pc1, pc2):
    pass


if __name__ == "__main__":

    # Some options
    target_scan_num = '1'
    igroup_sample_size = 500  # find this many points for each "strata"
    igroup_size = 5
    
    # sample acquisition options
    overlap_sample_size = 10000
    source_sample_size = 10000  # ???

    # Setup
    igroup_bounds = get_group_bounds(igroup_size)

    # Dataset path:
    datase_path = "150"  # better name?
    dataset_path = Path(dataset_path)
    neighborhoods_path = dataset_path / "neighborhoods"
    neighborhoods_path.mkdir(parents=True, exist_ok=True)
    
    plots_path = dataset_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    save_format = "{source_scan}_{target_scan}_{center}_{idx}.txt.gz"

    # Get point clouds ready to load in 
    scans_dir = Path("dublin/npy/")
    scan_paths = {f.stem:f.absolute() for f in scans_dir.glob("*.npy")}

    # Load target (reference) scan
    target_scan = np.load(scan_paths[target_scan_num])

    pbar = tqdm(scan_paths, total=len(scan_paths.keys()), dynamic_ncols=True)
    pbar.set_description("Total Progress")

    for source_scan_num in pbar:
        if source_scan is target_scan:
            continue  # skip
        
        # Load the next source scan        
        source_scan = np.load(scan_paths[source_scan_num])

        # build histogram - bins with lots of points in overlap likely exist
        #   in areas with a high concentration of other points in the overlap
        hist_info, _ = get_hist_overlap(target_scan, source_scan)

        for scan_info in zip(["ts", "ss"], [[target_scan, source_scan], [source_scan, source_scan]]):
            mode, scan_pairs = scan_info

            # obtain every point defined by the bin edges in `hist_info`
            aoi_idx = get_overlap_points(scan_pairs[0], hist_info)
            aoi = scan_pairs[1][aoi_idx] if mode is "ts" else scan_pairs[1][~aoi_idx]

            # save this information for future analysis
            plt.hist(aoi[:, 3], igroup_bounds)
            name = "Overlap" if "ts" else "Outside"
            plt.title("Dist. of Intensities for src/trgt: " 
                f"{source_scan_num}/{target_scan_num} - {name}.png")
            plt.savefig(str(plots_path / "{source_scan_num}_{target_scan_num}_{name}_i_dist.png"))

            # sample the points from the aoi 
            #   Note: no guarantee that there will be points in every strata 
            source_overlap_resampled = np.empty((0, source_overlap.shape[1]))

            for i, (l, h) in enumerate(igroup_bounds):
                strata = source_overlap[(intensities >= l) & (intensities < h)]
                
                