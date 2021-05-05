import code
import numpy as np
from pathlib import Path

# change this line to supply your own config!
from src.datasets.dublin.config import config
# from src.datasets.<my_dataset>.config import config

from src.datasets.tools.transforms import Corruption, GlobalShift
from src.config.pbar import get_pbar
from pptk import viewer


def harmonization_visualization(gt_collection_path, hm_collection_path, sample_size=100000, shift=False, view=True):
    gt_files = {f.stem:f for f in gt_collection_path.glob("*.npy")}
    hm_files = {f.stem:f for f in hm_collection_path.glob("*.npy")}
    
    collection = np.empty((len(hm_files) * sample_size, 11))
    C = Corruption(**config)
    G = GlobalShift(**config)

    pbar = get_pbar(
        enumerate(hm_files), 
        len(hm_files), 
        "loading collection",
        0, leave=True)

    for idx, scan_num in pbar:
        scan = np.load(gt_files[scan_num])
        if shift:
            scan = G(scan)
        if scan_num != config['target_scan']:
            corruption = C(scan)[1:, 3] # no copy of first point
            harmonization = np.load(hm_files[scan_num])[:, 3] # intensity only
            
            scan = np.concatenate((scan[:, :4], 
                                   np.expand_dims(corruption, 1),
                                   np.expand_dims(harmonization, 1),
                                   scan[:, 4:]), axis=1)
        else:
            scan = np.concatenate((scan[:, :4], 
                                   np.expand_dims(scan[:, 3], 1),
                                   np.expand_dims(scan[:, 3], 1),
                                   scan[:, 4:]), axis=1)

        sample = np.random.choice(len(scan), size=sample_size)
        collection[idx*sample_size:(idx+1)*(sample_size)] = scan[sample]

    
    if view:
        # view the collection

        # Center
        collection[:, :3] -= np.mean(collection[:, :3], axis=0)

        v = viewer(collection[:, :3])

        # show gt, corruption, and harmonization
        attr = [collection[:, 3], collection[:, 4], collection[:, 5]]
        v.color_map('jet', scale=[0, 1])
        v.attributes(*attr)
        mae = np.mean(np.abs(collection[:, 5] - collection[:, 3]))
        print("MAE: ", mae)

        # center the view
        v.set(lookat=[0,200,0], r=4250, theta=np.pi/2, phi=-np.pi/2)
        
        # remove background, axis, info
        v.set(bg_color=[1,1,1,1], show_grid=False, show_axis=False, show_info=False)

        # take photos
        v.set(curr_attribute_id=0)
        v.capture('gt.png')
        v.set(curr_attribute_id=1)
        v.capture('corruption.png')
        v.set(curr_attribute_id=2)
        v.capture('fix.png')

        code.interact(local=locals())

    return collection

if __name__ == "__main__":
    harmonization_visualization(config['scans_path'], 
                                config['harmonized_path'],
                                sample_size=100000,
                                shift=True)

