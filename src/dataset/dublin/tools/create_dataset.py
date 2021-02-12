import code
import logging
import logging.config
import numpy as np
from pathlib import Path
from pptk import kdtree
from tqdm import tqdm, trange
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import partial


from src.dataset.dublin.config import dublin_config

from src.dataset.tools.overlap import get_hist_overlap, get_overlap_points
from src.dataset.tools.overlap import neighborhoods_from_aoi, log_message

from src.dataset.dublin.tools.apply_rf import ApplyResponseFunction
from src.dataset.dublin.tools.shift import get_physical_bounds, apply_shift_pc


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
    return


if __name__ == "__main__":

    # Configuration
    target_scan_num = '1'
    config = dublin_config

    # Logging
    logging.config.fileConfig('tools/logging.conf')
    logger = logging.getLogger('datasetCreation')
    logger.info("Starting")

    # Neighborhoods path:
    neighborhoods_path = Path(config['save_path']) / "neighborhoods"
    neighborhoods_path.mkdir(parents=True, exist_ok=True)
    
    # Save neighborhoods as partial function
    save_format = "{source_scan}_{target_scan}_{center}_{idx}.txt.gz"
    func = partial(save_neighborhood, neighborhoods_path, save_format)

    # Get point clouds ready to load in
    scans_dir = Path("npy/")
    scan_paths = {f.stem:f.absolute() for f in scans_dir.glob("*.npy")}
    
    pbar = tqdm(scan_paths, total=len(scan_paths.keys()), dynamic_ncols=True)
    pbar.set_description("Total Progress [ | ]")

    # TO DO: target scan could be a collection of scans or tiles, so it will be
    #   necessary to check against a group of scans rather than just one. 
    #   Ideally, this variable would hold the largest segment of "target" 
    #   that is feasible. For dublin, it is simple to consider this as a single
    #   scan. However, to harmonize the entire region, harmonized scans will 
    #   become "target" scans. Futhermore, KY LiDAR and other tile-based LiDAR
    #   datasets will be another challenge in and of themselves.

    target_scan = np.load(scan_paths[target_scan_num])  

    for source_scan_num in pbar:
        if source_scan_num is target_scan_num:
            continue  # skip
        
        source_scan = np.load(scan_paths[source_scan_num])

        hist_info, _ = get_hist_overlap(target_scan, source_scan)

        # Overlap Region
        dsc = f"Total Progress [{target_scan_num}|{source_scan_num}]"
        pbar.set_description(dsc)
        aoi = target_scan[get_overlap_points(target_scan, hist_info)]

        overlap_size = neighborhoods_from_aoi(
            aoi,
            source_scan,
            "ts",
            (target_scan_num,source_scan_num),
            func,
            logger,
            **config)

        if overlap_size > config['min_overlap_size']:
            log_message("sufficent overlap, sampling outside", "INFO", logger)
            dsc = f"Total Progress [{source_scan_num}|{source_scan_num}]"
            pbar.set_description(dsc)
            aoi = source_scan[~get_overlap_points(source_scan, hist_info)]

            neighborhoods_from_aoi(
                aoi,
                source_scan,
                "ss",
                (target_scan_num,source_scan_num),
                func,
                logger,
                **config)
