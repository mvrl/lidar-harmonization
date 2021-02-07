import code
import urllib.request as request
import shutil
from contextlib import closing
from pathlib import Path
from multiprocessing import Pool
import time
# This script downloads KY Statewide lidar
# This takes around 5.5 hours on LCC with 8 processes
# versus ~2 days running wget -r URL

start_time = time.time()

def fetch_url(entry):
    path, uri = entry
    with closing(request.urlopen(uri)) as r:
        with open(path, 'wb') as f:
            shutil.copyfileobj(r, f)
    return path

# TO DO: Convert pg_src and output_path to CLI args
# TO DO: clean up list.txt to save space  -- done
# TO DO: test cleaned list.txt
pg_src = "list.txt"
output_path = Path("/home/dtjo223/workspace/lidar-harmonization/src/dataset/kylidar/laz")  # Fill in desired path here
URL = "ftp://ftp.kymartian.ky.gov/kyaped/LAZ/"
output_path.mkdir(exist_ok=True, parents=True)
files = []

# Parse

with open(pg_src, "r") as f:
    lines = f.readlines()

for filename in lines:
    files.append((str(output_path / filename).strip(), URL+filename))

print(f"Preparing to download {len(files)}")

# Download

pool = Pool(processes=8)
for path in pool.imap_unordered(fetch_url, files):
    print(path)

end_time = time.time()

print(f"Finished in {end_time - start_time:.3f} seconds")
