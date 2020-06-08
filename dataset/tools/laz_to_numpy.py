import time
import code
from pathlib import Path
import numpy as np
from laspy.file import File
from pptk import estimate_normals



def laz_to_np(path, include_scan_angle=True, keep_old=True):
    # convert folder of laz flights to .npy files for faster access
    append="_ang" if include_scan_angle else ""
    path = Path(path)
    files = [f for f in path.glob("*.laz")]

    for idx, f in enumerate(files):
        if Path(f"{f.parents[0] / (str(f.stem)+append)}.npy").exists() and keep_old:
            print(f"already processed {f.stem}, skipping")

        else:
            print(f"Opening {f}...")
        
            with File(f, mode="r") as laz:
                code.interact(local=locals()) 
                pts = np.stack([laz.x,
                                laz.y,
                                laz.z,
                                np.clip(laz.intensity, 0, 512)]).T

                if include_scan_angle:
                
                    normals = estimate_normals(pts[:,:3], 35, np.inf)
                    normals = normals * np.sign(normals[:, (2,)])
                    
                    pts = np.concatenate((pts,
                                      np.expand_dims(laz.scan_angle_rank, 1),
                                      normals), axis=-1)
                
                print(f"Saving new numpy file as {f.parents[0] / f.stem}.npy")
                np.save(f"{f.parents[0] / (str(f.stem)+append)}.npy", pts)

if __name__ == '__main__':
    start_time = time.time()
    laz_to_np('dublin_flights')
    print(f"Finished in {time.time() - start_time} seconds")
        
    
    
