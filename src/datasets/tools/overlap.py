import code
import numpy as np
from pptk import kdtree
from pathlib import Path
import matplotlib.pyplot as plt
from multiprocessing import Pool
import sharedmem
from functools import partial
import h5py

from src.config.pbar import get_pbar
from tqdm import tqdm, trange

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

def get_indices(pc, data):
    # Crazy idea/TODO: don't need to be so accurate since we have to check
    #   point validity anyway. Just take all bins with c > 1 and find
    #   the largest square that encompasses that area. 
    x1, x2, y1, y2, z1, z2 = data
       
    # this is very slow :/
    new_indices = ((x1 <= pc[:, 0]) & (pc[:, 0] < x2) &
                   (y1 <= pc[:, 1]) & (pc[:, 1] < y2) &
                   (z1 <= pc[:, 2]) & (pc[:, 2] < z2))
        
    return new_indices

def get_overlap_points(pc, hist_info, config, c=1, pb_pos=1):
    # Pull points out of `pc` from overlap information to be used in dataset
    # creation.
    #   `hist_info`: tuple (hist, bins) 
    #   `c` : the count of required overlap points to exist in a bin for it to
    #       to count as being "in the overlap." Higher values of c grab points 
    #       more likely to be in the overlap.

    
    indices = np.full(pc.shape[0], False, dtype=bool)
    process_list = []
    hist, (xedges, yedges, zedges) = hist_info
    my_func = partial(get_indices, pc)
    workers = config['workers']
    
    # if str(type(pc)) != "<class 'sharedmem.sharedmem.anonymousmemmap'>":
    #     exit("ERROR : PC isn't in shared memory!")
    
    h_iter = np.array(np.meshgrid(
        np.arange(hist.shape[0]), 
        np.arange(hist.shape[1]),
        np.arange(hist.shape[2])
    )).T.reshape(-1, 3)
    
    pbar = get_pbar(
        h_iter,
        len(h_iter),
        "Building Processes", 
        pb_pos, disable=config['tqdm'])

    for t in pbar:
        i, j, k = t
        if hist[i][j][k] > c:
            x1, x2 = xedges[i], xedges[i+1]
            y1, y2 = yedges[j], yedges[j+1]
            z1, z2 = zedges[k], zedges[k+1]
            process_list.append((x1, x2, y1, y2, z1, z2))

            
    # multiprocessing - this is maybe 2x as fast with 8 workers?
    with Pool(workers) as p: 
        sub_pbar = get_pbar(
            p.imap_unordered(my_func, process_list),
            len(process_list),
            "Querying AOI",
            pb_pos, disable=config['tqdm']
            )

        for new_indices in sub_pbar:
            indices = indices | new_indices

    # single threaded
    # for t in tqdm(process_list, desc="  "*pb_pos + "Querying AOI", position=pb_pos, leave=False):
    #     indices = indices | get_indices(t)
    
    return indices

def get_igroup_bounds(bin_size):
    return [(i, i+bin_size) for i in np.linspace(0, 1-bin_size, int(1/bin_size))]

def filter_aoi(kd, aoi, config, pb_pos=1):
    # Querying uses a large amount of memory, use chunking to keep 
    #   the footprint small
    keep = []
    max_chunk_size = config['max_chunk_size']

    max_idx = int(np.ceil(aoi.shape[0] / max_chunk_size))
    sub_pbar = get_pbar(
            range(0, aoi.shape[0], max_chunk_size),
            max_idx,
            "Filtering AOI",
            pb_pos, disable=config['tqdm']
            )

    for i in sub_pbar:
        current_chunk = aoi[i:i+max_chunk_size]
        query = kdtree._query(kd, 
                              current_chunk[:, :3], 
                              k=config['max_n_size'], dmax=1)


        for j in range(len(query)):
            if len(query[j]) == config['max_n_size']:
                keep.append(i+j)

    return aoi[keep]


def save_neighborhoods_hdf5_eval(aoi, query, source_scan, config, chunk_size=500, pb_pos=2):
    with h5py.File(config['eval_dataset'], "a") as f:

        start = f['eval'].shape[0]
        end = start + aoi.shape[0]
        slices = (slice(start, end), slice(0, config['max_n_size']+1), slice(0, 9))

        f['eval'].resize(end, axis=0)        

        sub_pbar = get_pbar(
            f['eval'].iter_chunks(sel=slices),
            int(np.ceil(aoi.shape[0]/chunk_size)),
            "Saving Neighborhoods",
            pb_pos, disable=config['tqdm'], leave=True)

        for idx, chunk in enumerate(sub_pbar):
            aoi_chunk = aoi[chunk[0].start-start:chunk[0].stop-start]
                        
            query_chunk = query[chunk[0].start-start:chunk[0].stop-start]

            aoi_chunk = np.expand_dims(aoi_chunk, 1)
            
            neighborhoods = np.concatenate(
                (aoi_chunk, source_scan[query_chunk]),
                axis=1)
            f['eval'][chunk] = neighborhoods


def save_neighborhoods_hdf5(aoi, query, source_scan, config, chunk_size=500, pb_pos=2):
    with h5py.File(config['dataset_path'], "a") as f:
        # the goal is to load as little of this into memory at once
        aoi_ = {}; query_ = {}
        train_idx = np.full(len(aoi), False, dtype=bool)
        train_size = int(config['splits']['train']*aoi.shape[0])

        # set some percentage for training, shuffle.
        train_idx[:train_size] = True; np.random.shuffle(train_idx)

        aoi_['train'] = aoi[train_idx]
        aoi_['test'] = aoi[~train_idx]
        query_['train'] = query[train_idx]
        query_['test'] = query[~train_idx]

        chunk_size = f['train'].chunks[0]  # this is the same for test/train
        sub_pbar = get_pbar(
            list(f.keys()),
            len(list(f.keys())),
            "Saving Neighborhoods []",
            pb_pos, disable=config['tqdm'])

        for split in sub_pbar:
            start = f[split].shape[0]
            end = start + aoi_[split].shape[0]
            slices = (slice(start, end), slice(0, config['max_n_size']+1), slice(0, 9))

            f[split].resize(end, axis=0)
            sub_pbar2 = get_pbar(
                f[split].iter_chunks(sel=slices),
                int(np.ceil(aoi_[split].shape[0]/chunk_size)),  # this could be more clear
                "Saving chunks",
                pb_pos+1, disable=config['tqdm'])
            
            # indices for the h5 chunks begin at start
            # indices for aoi and query begin at 0
            for idx, chunk in enumerate(sub_pbar2):
                # chunk is a tuple of slices with each element corresponding to
                #   a dimension of the dataset. 0 is the first axis. 
                aoi_chunk = aoi_[split][chunk[0].start-start:chunk[0].stop-start]
                aoi_chunk = np.expand_dims(aoi_chunk, 1)

                query_chunk = query_[split][chunk[0].start-start:chunk[0].stop-start]
                neighborhoods = np.concatenate((
                    aoi_chunk, source_scan[query_chunk]),
                    axis=1)

                f[split][chunk] = neighborhoods


def resample_aoi(aoi, igroup_bounds, max_size, config, pb_pos=2):
    # We want to resample the intensities here to be balanced
    #   across the range of intensities. 
    aoi_resampled = np.empty((0, aoi.shape[1]))
    sub_pbar = get_pbar(
        igroup_bounds,
        len(igroup_bounds),
        "Resampling AOI",
        pb_pos, disable=config['tqdm'])

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
        config,
        logger=None):

    # TO DO: this is pretty tightly coupled and needs to be refactored
    #   so that logging and checking overlap sizes can be more cohesive.
    igroup_bounds = get_igroup_bounds(config['igroup_size'])
    t_num, s_num = scan_nums

    # Build kd tree over source scan
    kd = kdtree._build(source_scan[:, :3])

    # filter out target points from aoi with neighborhoods in source 
    #   with size < `max_n_size`
    aoi = filter_aoi(
        kd,
        aoi,
        config,
        pb_pos=2)

    overlap_size = aoi.shape[0]

    log_message(f"[{t_num}|{s_num}][{mode}] overlap size post-filter: {aoi.shape}", "INFO", logger)

    if overlap_size >= config['min_overlap_size']:
        bins = [i[0] for i in igroup_bounds] + [igroup_bounds[-1][1]]

        plot_hist(aoi, bins, mode+"-B", 
            s_num, t_num, config['plots_path'])

        # resample aoi
        aoi = resample_aoi(
            aoi, 
            igroup_bounds, 
            config['igroup_sample_size'],
            config)

        log_message(f"[{t_num}|{s_num}][{mode}] overlap size post-resample: {aoi.shape}", "INFO", logger)

        # Verify resampling operation
        plot_hist(aoi, bins, mode+"-P", 
            s_num, t_num, config['plots_path'])

        # Query neighborhoods from filtered resampled aoi
        query = kdtree._query(kd, 
                              aoi[:, :3], 
                              k=config['max_n_size'], dmax=1)

        query = np.array(query).astype(np.int)

        save_neighborhoods_hdf5(aoi, query, source_scan, config)

    return overlap_size 
