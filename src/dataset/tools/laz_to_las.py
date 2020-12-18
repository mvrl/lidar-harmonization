import time
from pathlib import Path
import sys
from multiprocessing import Pool
import subprocess

las_dir = Path("/home/dtjo223/workspace/kylidar/las_raw")
las_dir.mkdir(exist_ok=True, parents=True)

def laz_to_las(path):
    global las_dir
    new_path = las_dir / (path.stem + '.las')
    process = subprocess.Popen(f"laszip -i {path} -o {new_path}".split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

if __name__ == "__main__":

    start_time = time.time()
    p = Pool(24)

    laz_dir = Path("/home/dtjo223/workspace/kylidar/laz_raw")
    files = laz_dir.glob("*.laz")

    for idx, _ in enumerate(p.imap_unordered(laz_to_las, files)):
        print(idx)

    end_time = time.time()

    print(f"Finished in {end_time - start_Time}")
