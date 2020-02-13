import time
from pathlib import Path
import numpy as np
from laspy.file import File


def laz_to_np(path):
    # convert folder of laz flights to .npy files for faster access
    path = Path(path)
    files = [f for f in path.glob("*.laz")]

    for idx, f in enumerate(files):
        print(f"Opening {f}...")
        laz = File(f)

        # dublin intensities spikes to 2^16 but primarily fall within
        # 2^12
        npy = np.stack([laz.x,
                        laz.y,
                        laz.z,
                        np.clip(laz.intensity, 0, 512)]).T
        print(f"Saving new numpy file as {f.parents[0] / f.stem}.npy")
        np.save(f"{f.parents[0] / f.stem}.npy", npy)

if __name__ == '__main__':
    start_time = time.time()
    laz_to_np('dublin_flights')
    print(f"Finished in {time.time() - start_time} seconds")
        
    
    
