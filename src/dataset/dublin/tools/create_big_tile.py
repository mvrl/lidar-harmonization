import time
import code
import json
import numpy as np
from src.dataset.tools.metrics import create_kde

from pathlib import Path
from pptk import kdtree, viewer
import matplotlib.pyplot as plt
from src.dataset.dublin.tools.apply_rf import ApplyResponseFunction
from src.dataset.dublin.tools.shift import get_physical_bounds, apply_shift_pc


def load_laz(path):
    f = np.load(path)
    return f

def visualize_n_flights(path, flights, sample_size=500000, shift=False):
    # flights is a list of numbers representing flights 0-40
    start_time = time.time()
    laz_files_path = Path(path)
    bounds = get_physical_bounds(scans=path, bounds_path="bounds.npy")
    
    pts = np.empty((0, 9))
    for scan in flights:
        
        fi = np.load(laz_files_path / (str(scan)+".npy"))
        if sample_size:
            sample = np.random.choice(len(fi), size=sample_size)
            pts = np.concatenate((pts, fi[sample])) 
        else:
            pts = np.concatenate((pts, fi))

    if shift:
        pts = apply_shift_pc(pts, bounds[0][0], bounds[0][1])
    attr1 = pts[:, 3]
    attr2 = pts[:, 8]

    v = viewer(pts[:, :3])
    v.attributes(attr1, attr2)
    # code.interact(local=locals())


def create_big_tile_manual(path, base_flight, intersecting_flight, manual_point, num_points=1e6, in_overlap=True, shift=True):

    # NOTE: shift does not support in_overlap currently

    laz_files_path = Path(path)
    ARF = ApplyResponseFunction("dorf.json", "mapping.npy")
    bounds = get_physical_bounds(scans=path, bounds_path="bounds.npy")

    if in_overlap:
        save_dir = Path("synth_crptn/big_tile_in_overlap")
    else:
        save_dir = Path("synth_crptn/big_tile_no_overlap")

    save_dir.mkdir(parents=True, exist_ok=True)

    if shift:
        shift_save_dir = Path("synth_crptn+shift/big_tile_no_overlap")
        shift_save_dir.mkdir(parents=True, exist_ok=True)

    
    # if we are in the overlap, we have to process two kd trees, 
    # otherwise we can just do 1. 
    if in_overlap:
        flight1 = np.load(laz_files_path / (str(base_flight)+".npy"))
        print(f"Loaded flight {flight1[0, 8]}")
        kd1 = kdtree._build(flight1[:, :3])
        q1 = kdtree._query(kd1, manual_point, k=1e6)
        tile_f1 = flight1[tuple(q1)]
    flight2 = np.load(laz_files_path / (str(intersecting_flight)+".npy"))
    print(f"Loaded flight {flight2[0, 8]}")
    
    kd2 = kdtree._build(flight2[:, :3])
    q2 = kdtree._query(kd2, manual_point, k=1e6)
    tile_f2 = flight2[tuple(q2)]

    if in_overlap:
        tile_f2_1 = tile_f2.copy()
        tile_f1_2 = tile_f1.copy()
        tile_f2_1[:, 8] = float(base_flight)
        tile_f1_2[:, 8] = float(intersecting_flight)
     
        # threshold the axis values to create a "clean" divide
        # 0 = x axis, 1 = y axis, 2 = z axis (??)
        my_axis=0
        tile_f1 = tile_f1[np.logical_not(tile_f1[:, my_axis] <= manual_point[0, my_axis])]
        tile_f2 = tile_f2[np.logical_not(tile_f2[:, my_axis] > manual_point[0, my_axis])]

        # Fill in missing points from overlap
        tile_f2_1 = tile_f2_1[np.logical_not(tile_f2_1[:, my_axis] <= manual_point[0, my_axis])]
        tile_f1_2 = tile_f1_2[np.logical_not(tile_f1_2[:, my_axis] > manual_point[0, my_axis])]
        tile_f1 = np.concatenate((tile_f1, tile_f2_1))
        tile_f2 = np.concatenate((tile_f2, tile_f1_2))

    # create corrupted tile_f2
    tile_f2_alt = ARF(tile_f2, intersecting_flight, 512) 

    if shift:
        tile_f2_shift = apply_shift_pc(tile_f2.copy(), bounds[0][0], bounds[0][1])
        tile_f2_alt_shift = ARF(tile_f2_shift.copy(), intersecting_flight, 512)

   
    sample = np.random.choice(len(tile_f2), size=5000)
    create_kde(
            tile_f2[sample][:, 3],
            tile_f2_alt[sample][:, 3],
            "ground truth",
            "altered values",
            save_dir / "post_response_curve.png")
            
    print("Created post response curve kde")
    
    if in_overlap:
        tile_gt = np.concatenate((tile_f1, tile_f2))
        tile_alt = np.concatenate((tile_f1, tile_f2_alt))
        attr1 = tile_gt[:, 3]
        attr2 = tile_alt[:, 8]
        attr3 = tile_alt[:, 3]
        v = viewer(tile_gt[:, :3])

    else:
        attr1 = tile_f2[:, 3]
        attr2 = tile_f2[:, 8]
        attr3 = tile_f2_alt[:, 3]
        v = viewer(tile_f2[:, :3])
        if shift:
            attr4 = tile_f2_shift[:, 3]
            attr5 = tile_f2_alt_shift[:, 3]
            v.attributes(attr1, attr2, attr3, attr4, attr5)
            print("gt, alt, scan #, shift gt, shift alt")
        else:
            v.attributes(attr1, attr2, attr3)

    v.set(lookat=manual_point[0])

    if in_overlap:

        np.save(save_dir / "base_flight_tile.npy", tile_f1)

    np.savetxt(save_dir / "gt.txt.gz", tile_f2)
    np.savetxt(save_dir / "alt.txt.gz", tile_f2_alt)
    if shift:
        np.savetxt(shift_save_dir / "gt.txt.gz", tile_f2_shift)
        np.savetxt(shift_save_dir / "alt.txt.gz", tile_f2_alt_shift)

    print("tiles saved")
    code.interact(local=locals())

if __name__=='__main__':
    
    # Show the overlapping flights over flight 1
    # visualize_n_flights('dublin/npy', [1, 39], sample_size=None, shift=True)
    
    # Create a big tile with manual point input in the overlap region
    # shifting -100 X units places this within the transition zone
    create_big_tile_manual(
        'dublin/npy', 
        1,   # Target scan
        39,  # Source scan
        np.array([[316120.0, 234707.422, 1.749]]), # center of AOI
        in_overlap=False, shift=True)
    


    
                 
    
