# This script will use ordinary histogram matching to harmonize each tile to 
# the reference distribution. 

import code
import numpy as np
from pathlib import Path
from laspy.file import File
from src.evaluation.histogram_matching import hist_match
from multiprocessing import Pool

def structure_data(las):
    return np.stack((
        las.x, las.y, las.z, las.intensity)).T

# TO DO: make these into CL args
ky_lidar_raw_files_path = Path("/home/dtjo223/workspace/kylidar/las_berea_test")
ky_lidar_fix_path = Path("/home/dtjo223/workspace/kylidar/las_berea_test_harmonized")
ky_lidar_fix_path.mkdir(parents=True, exist_ok=True)
# get list of files to harmonize
ky_lidar_raw_files_paths = list(ky_lidar_raw_files_path.glob("*.las"))

# How best to choose the reference tile? Maybe just choose the first one for now?
ref_tile_path = ky_lidar_raw_files_paths.pop(0)

# I don't know of a good way to mark it so that it's visible later though...
print(f"Reference tile: {ref_tile_path}")

# Open ref tile, save it as-is in the new path
ref_tile = File(ref_tile_path, mode='r')
new_tile = File(
    ky_lidar_fix_path / (ref_tile_path.stem + ref_tile_path.suffix),
    mode='w',
    header=ref_tile.header)

new_tile.intensity = ref_tile.intensity
new_tile.close()

# need a function to process a sample and save it back
def do_hm(target_tile_path):
    # load tile for matching
    target_tile_name = target_tile_path.stem + target_tile_path.suffix
    try:
        target_tile = File(target_tile_path, mode='r') # this can take some time
     
        # use global to get the info, apologize to programming teachers
        global ref_tile
        global ky_lidar_fix_path
    
        # match histograms
        new_dist = hist_match(target_tile.intensity, ref_tile.intensity)
	
        # apply transformation
        new_tile_path = ky_lidar_fix_path / target_tile_name
        new_tile = File(new_tile_path, mode='w', header=target_tile.header)
        new_tile.points = target_tile.points
        new_tile.intensity = new_dist
        new_tile.close()
        return (True, new_tile_path)
    
    except:
        # if we can't load the file, that's a big problem, but it seems to happen
        # sometimes. 
        return (False, target_tile_path)


pool = Pool(24)

for result in pool.imap(do_hm, ky_lidar_raw_files_paths):
    s, path = result 
    if s is True:
         print(f"Success! {path}")
    else:
         print(f"--- ERROR: {path}")
    
