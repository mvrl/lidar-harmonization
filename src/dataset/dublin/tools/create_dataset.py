import code
import gc
import numpy as np
from pathlib import Path
from pptk import kdtree
from tqdm import tqdm, trange
from multiprocessing import Pool
import matplotlib.pyplot as plt
from functools import partial


from src.dataset.dublin.config import dublin_config

from src.dataset.tools.overlap import get_hist_overlap, get_overlap_points
from src.dataset.tools.overlap import neighborhoods_from_aoi

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

    for source_scan_num in pbar:
        if source_scan_num is target_scan_num:
            continue  # skip

        for mode in ["ss", "ts"]:

            neighborhoods_from_aoi(
                source_scan_num, 
                target_scan_num,
                scan_paths,
                mode,
                func,
                pbar,
                **config)
