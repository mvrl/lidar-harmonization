import time
import code
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from laspy.file import File
from pptk import estimate_normals

from patch.patch import patch_mp_connection_bpo_17560

def build_np(args):
    path, idx = args
    with File(path, mode='r') as laz:
        pt_src_id = laz.pt_src_id.copy()
        pt_src_id.fill(idx)

        pts = np.stack([laz.x,
                        laz.y,
                        laz.z,
                        np.clip(laz.intensity, 0, 512),
                        laz.scan_angle_rank]).T

        return pts, pt_src_id, path


def laz_to_np(path):
    # convert folder of laz flights to .npy files for faster access
    path = Path(path)
    files = [f for f in path.glob("laz/*.laz")]
    indices = np.arange(0, len(files))
    save_path = path / 'npy'
    save_path.mkdir(parents=True, exist_ok=True)
    for data in zip(files, indices):
        
        pts, pt_src_id, f = build_np(data)
        normals = estimate_normals(pts[:,:3], 35, np.inf, num_procs=9)
        normals = normals * np.sign(normals[:, (2,)])

        pts = np.concatenate((pts,
                              normals,
                              np.expand_dims(pt_src_id, 1)), axis=-1)

        # this should provide some guarantee that each flight is what 
        # we are expecting it to be
        flight_num = str(int(pts[0][8]))
        
        # flights are now instantly identifiable by filepath as well as 
        # their pt_src_id attribute 
        print(f"Saving new numpy file as {f.parents[1] / 'npy' / flight_num}.npy")
        np.save(f"{f.parents[1] / 'npy' / flight_num}.npy", pts)

if __name__ == '__main__':
    start_time = time.time()
    laz_to_np('dublin')
    print(f"Finished in {time.time() - start_time} seconds")
        
    
    
