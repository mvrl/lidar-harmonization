import time
import code
import numpy as np
import pandas as pd
from pathlib import Path
from pptk import kdtree
from tqdm import tqdm, trange
from src.dataset.tools.apply_rf import ApplyResponseFunction
from src.dataset.tools.shift import get_physical_bounds, apply_shift_pc

def get_igroup_bounds(bin_size):
        return [(i, i+bin_size) for i in range(0, 512, bin_size)]

def get_hist_overlap(pc1, pc2, s=10000, hist_bin_length=10):
    # build a 2d histogram over the overlap region. Each bin contains the 
    #   frequency of points
    
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
    # Pull points out of `pc` from overlap information to be used in dataset
    # creation.
    #   `hist_info`: tuple (hist, bins) 
    #   `c` : the count of required overlap points to exist in a bin for it to
    #       to count as being "in the overlap." Higher values of c grab points 
    #       more likely to be in the overlap. This can be exploited by supplying
    #       a low value to find points outside the overlap by using 
    #       `np.delete(pc, indices)`.
    #
    # this seems slow... 
    
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

def build_neighborhoods(pc1, pc2):
    pass


if __name__ == "__main__":

    # Some options
    target_scan_num = '1'
    shift = True
    igroup_sample_size = 500  # not sure how many will be needed
    igroup_size = 5
    
    # sample acquisition options
    overlap_sample_size = 10000
    source_sample_size = 10000  # ???
    overlap_threshold=150

    # other?
    workers=6
    
    # Setup
    ARF = ApplyResponseFunction("dorf.json", "mapping.npy")
    bounds = get_physical_bounds(scans="dublin/npy", bounds_path="bounds.npy")
    igroup_bounds = get_group_bounds(igroup_size)

    # Dataset path:
    datase_path = "synth_crptn+shift" if shift else "synth_crptn"
    dataset_path = Path(dataset_path)
    neighborhoods_path = dataset_path / "150" / "neighborhoods"
    neighborhoods_path.mkdir(parents=True, exist_ok=True)
    
    plots_path = dataset_path / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    save_format = "{source_scan}_{target_scan}_{center}_{idx}.txt.gz"

    # Get point clouds ready to load in 
    scans_dir = Path("dublin/npy/")
    scan_paths = {f.stem:f.absolute() for f in pc_dir.glob("*.npy")}

    # Load target (reference) scan
    target_scan = np.load(pc_paths[target_scan_num])

    pbar = tqdm(pc_paths, total=len(scan_paths.keys()), dynamic_ncols=True)
    pbar.set_description("Total Progress")

    for source_scan_num in pbar:
        if source_scan is target_scan:
            continue  # skip
        
        # Load the next source scan        
        source_scan = np.load(pc_paths[source_scan_num])

        # build histogram - bins with lots of points in overlap likely exist
        #   in areas with a high concentration of other points in the overlap
        hist_info, _ = get_hist_overlap(target_scan, source_scan)

        # obtain every point defined by the bin edges in `hist_info`
        overlap_indices = get_overlap_points(
            target_scan, 
            hist_info, 
            overlap_threshold)

        pc_overlap = pc1[overlap_indices]

        # pc_overlap likely contains many points, but the distribution of 
        #   intensities in the dataset needs to be uniform. To accomplish this,
        #   it is easiest to simply pick `igroup_sample_size` points from each
        #   strata of intensity values desired. 

        intensities = pc_overlap[:, 3]
        hist, _ = np.histogram(intensities, igroup_bounds)

        # save this information for future analysis
        plt.hist(intensities, igroup_bounds)
        plt.title("Dist. of Intensities for src/trgt: " 
            f"{source_scan_num}/{target_scan_num}.png")
        plt.savefig(str(plots_path))

        # resample the points from the overlap region. Unfortunately,
        #   there is no guarantee that there will be points in every strata 
        #   (especially in the global shift version of this dataset).

        pc_overlap_resampled = np.empty((0, pc_overlap.shape[1]))

        for i, (l, h) in enumerate(igroup_bounds):
            curr_strata = pc_overlap[(intensities >=) l & (intensities < h)]
            if len(curr_strata):
                pc_overlap_resampled = np.concatenate((
                    pc_overlap_resampled,
                    curr_strata[
                        np.random.choice(len(curr_strata), igroup_sample_size)
                        ]
                    ))

        # need to build another region from just the source. The below lines
        #   generate a histogram of overlapping points from source into target,
        #   then the points that do not overlap are pulled out.
        hist_info, _ = get_hist_overlap(source_scan, target_scan)

        overlap_indices = get_overlap_points(
            source_scan,
            hist_info,
            overlap_threshold,  # this is ignored
            invert=True
            )

        source_nonoverlap = pc2[overlap_indices]
                
        for i, (l, h) in enumerate(igroup_bounds):
            curr_strata = pc_






        

