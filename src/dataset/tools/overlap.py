import code
import numpy as np
from pptk import kdtree
from tqdm import tqdm, trange
from pathlib import Path
import matplotlib.pyplot as plt
from multiprocessing import Pool


def get_hist_overlap(pc1, pc2, sample_overlap_size=10000, hist_bin_length=25):
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
    
    bin_counts = [int((f[1]-f[0])/hist_bin_length) for f in data_range]

    del pc_combined  # save some mem
    
    # define bins based on data_range:
    x_bins = np.linspace(data_range[0][0], data_range[0][1], num=bin_counts[0])
    y_bins = np.linspace(data_range[1][0], data_range[1][1], num=bin_counts[1])
    z_bins = np.linspace(data_range[2][0], data_range[2][1], num=bin_counts[2])
    
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
        bins=[x_bins, y_bins, z_bins])

    return (hist, edges), pc1_sample_f

def get_overlap_points(pc, hist_info, c=1):
    # Pull points out of `pc` from overlap information to be used in dataset
    # creation.
    #   `hist_info`: tuple (hist, bins) 
    #   `c` : the count of required overlap points to exist in a bin for it to
    #       to count as being "in the overlap." Higher values of c grab points 
    #       more likely to be in the overlap.

    
    indices = np.full(pc.shape[0], False, dtype=bool)
    process_list = []
    hist, (xedges, yedges, zedges) = hist_info
    
    def get_indices(e):
        x1, x2, y1, y2, z1, z2 = e
        
        # this is very slow :/
        new_indices = ((x1 <= pc[:, 0]) & (pc[:, 0] < x2) &
                       (y1 <= pc[:, 1]) & (pc[:, 1] < y2) &
                       (z1 <= pc[:, 2]) & (pc[:, 2] < z2))
        
        return new_indices
        
    h_iter = np.array(np.meshgrid(
        np.arange(hist.shape[0]), 
        np.arange(hist.shape[1]),
        np.arange(hist.shape[2])
    )).T.reshape(-1, 3)
    
    for t in tqdm(h_iter, desc="Building processes", position=1, leave=False):
        i, j, k = t
        if hist[i][j][k] > c:
            x1, x2 = xedges[i], xedges[i+1]
            y1, y2 = yedges[j], yedges[j+1]
            z1, z2 = zedges[k], zedges[k+1]
            process_list.append((x1, x2, y1, y2, z1, z2))
            
    process_list = np.array(process_list)
    for t in tqdm(process_list, desc="  Querying AOI", position=1, leave=False):
        indices = indices | get_indices(t)
    
    return indices

def get_igroup_bounds(bin_size):
        return [(i, i+bin_size) for i in range(0, 512, bin_size)]

def filter_aoi(kd, aoi, max_chunk_size, max_n_size, pb_pos=1):
    # Querying uses a large amount of memory, use chunking to keep 
    #   the footprint small
    keep = []
    curr_idx = 1; max_idx = np.ceil(aoi.shape[0] / max_chunk_size)
    sub_pbar = trange(0, aoi.shape[0], max_chunk_size,
                      desc="  Filtering AOI", leave=False, position=pb_pos)
    for i in sub_pbar:
        current_chunk = aoi[i:i+max_chunk_size]
        query = kdtree._query(kd, 
                              current_chunk[:, :3], 
                              k=max_n_size, dmax=1)

        sub2_pbar = tqdm(range(len(query)),
                    desc=f"    Filtering [{curr_idx}/{max_idx}]",
                    leave=False,
                    position=pb_pos+1,
                    total=len(query))
    
        for j in sub2_pbar:
            if len(query[j]) == max_n_size:
                keep.append(i+j)

        curr_idx+=1

    return aoi[keep]


def save_neighborhoods(aoi, query, source_scan, save_func, workers=8, chunk_size=5000, pb_pos=1):
    # Indexing query into source_scan is expensive, as this returns a 
    #    [N, 150, 9] array as a copy. Chunking can save considerable memory in 
    #    this case, which can prevent undesired terminations. Not sure what a 
    #    reasonable chunk size might be. AOI is typically 50k, so maybe ~5k?

    curr_idx = 1; max_idx = np.ceil(aoi.shape[0] / chunk_size)
    sub_pbar = trange(0, aoi.shape[0], chunk_size,
                      desc=f"  Saving Neighborhoods", leave=False, position=pb_pos)

    for i in sub_pbar:
        aoi_chunk = aoi[i:i+chunk_size, :]
        query_chunk = query[i:i+chunk_size, :]

        aoi_chunk = np.expand_dims(aoi_chunk, 1)
        neighborhoods = np.concatenate(
            (aoi_chunk, source_scan[query_chunk]),
            axis=1)

        data = zip(range(i, i+neighborhoods.shape[0]), neighborhoods)


        with Pool(workers) as p:
            sub_pbar2 = tqdm(p.imap_unordered(save_func, data),
                desc=f"    Processing Chunk [{curr_idx}/{max_idx}]",
                total=neighborhoods.shape[0],
                position=pb_pos+1,
                leave=False)

            for _ in sub_pbar2:
                pass

        curr_idx+=1


def resample_aoi(aoi, igroup_bounds, max_size, pb_pos=1):
    # We want to resample the intensities here to be balanced
    #   across the range of intensities. 
    aoi_resampled = np.empty((0, aoi.shape[1]))
    sub_pbar = tqdm(igroup_bounds,
                    desc="  Resampling AOI",
                    leave=False,
                    position=pb_pos,
                    total=len(igroup_bounds))

    for (l, h) in sub_pbar:
        strata = aoi[(l <= aoi[:, 3]) & (aoi[:, 3] < h)]
        # auto append if strata is small
        if strata.shape[0] <= max_size:
            aoi_resampled = np.concatenate((
                aoi_resampled, strata))

        # random sample if large
        else:
            sample = np.random.choice(len(strata), max_size)
            aoi_resampled = np.concatenate((
                aoi_resampled, strata[sample]))

    return aoi_resampled

def plot_hist(aoi, bins, mode, source_scan_num, target_scan_num, save_path):
    # mode is a string in the form of "XX-Y" where XX is "ts" or "ss" for target-source
    #   or source-source and Y is "P" for post-sampling or "B" for pre-sampling
    plots_path = Path(save_path) / "plots"
    plots_path.mkdir(parents=True, exist_ok=True)

    XX, Y = mode.split("-")
    if XX not in ["ts", "ss"]:
        exit(f"Wrong format code: {mode}")
    if Y not in ["P", "B"]:
        exit(f"Wrong f ormat code: {mode}")

    m = "Overlap" if XX == "ts" else "Outside"
    n = "Pre" if Y == "B" else "Post"

    plt.hist(aoi[:, 3], bins)
    n = "verlap" if XX is "ts" else "Outside"

    plt.title(f"{n}-Sample Dist. of Intensities for src/trgt: "
        f"{source_scan_num}/{target_scan_num} - {m}.png")
    fname = f"{source_scan_num}_{target_scan_num}_{m}_{n.lower()}_i_dist.png"
    plt.savefig(str(plots_path / fname))
    plt.close()

def log_message(msg, level, logger=None):
    # level can be "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    if logger and level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        if level is "DEBUG":
            logger.debug(msg)
        if level is "INFO":
            logger.info(msg)
        if level is "WARNING":
            logger.warning(msg)
        if level is "CRITICAL":
            logger.critical(msg)
        return
    if level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        exit("logger level not understood:", level)
    if not logger:
        return

def neighborhoods_from_aoi(
        aoi,
        source_scan,
        mode,
        scan_nums,
        save_func,
        logger=None,
        **kwargs):

    igroup_bounds = get_igroup_bounds(kwargs['igroup_size'])
    t_num, s_num = scan_nums

    # Build kd tree over source scan
    kd = kdtree._build(source_scan[:, :3])

    # filter out target points from aoi with neighborhoods in source 
    #   with size < `max_n_size`
    aoi = filter_aoi(
        kd, 
        aoi, 
        kwargs['max_chunk_size'], 
        kwargs['max_n_size'])

    overlap_size = aoi.shape[0]

    log_message(f"[{t_num}|{s_num}][{mode}] overlap size post-filter: {aoi.shape}", "INFO", logger)

    if overlap_size >= kwargs['min_overlap_size']:

        bins = [i[0] for i in igroup_bounds] + [igroup_bounds[-1][1]]
        plot_hist(aoi, bins, mode+"-B", 
            s_num, t_num, kwargs['save_path'])

        # resample aoi
        aoi = resample_aoi(
            aoi, 
            igroup_bounds, 
            kwargs['igroup_sample_size'])

        log_message(f"[{t_num}|{s_num}][{mode}] overlap size post-resample: {aoi.shape}", "INFO", logger)


        # Verify resampling operation
        plot_hist(aoi, bins, mode+"-P", 
            s_num, t_num, kwargs['save_path'])

        # Query neighborhoods from filtered resampled aoi
        query = kdtree._query(kd, 
                              aoi[:, :3], 
                              k=kwargs['max_n_size'], dmax=1)

        query = np.array(query).astype(np.int)

        save_neighborhoods(aoi, query, source_scan, save_func)

    return overlap_size 
