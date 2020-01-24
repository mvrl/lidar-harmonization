import time
import code
import numpy as np
import json
from pathlib import Path

start_time = time.time()

# Read in the response functions file
with open('dataset/response_functions.json') as json_file:
    data = json.load(json_file)

print(data.keys())  # names scales brightness intensities

# get dataset file directory ready
dataset_path = Path(r"/home/david/bin/python/dublin/dataset/gt")
files = [file for file in dataset_path.glob('*.npy')]
file_count = len(files)

print("found %s files to alter" % file_count)

for file_path in files:
    file_name = file_path.stem
    flight_num, index = file_name.split('_')[0], file_name.split('_')[1]
    

    curr_sample = np.load(file_path)
    curr_intensities = curr_sample[:,3]
    altered_intensities = np.interp(curr_intensities,
                                    np.array(data['brightness'][str(flight_num)])*255,
                                    np.array(data['intensities'][str(flight_num)])*255)

    altered_sample = np.copy(curr_sample)
    altered_sample[:,3] = altered_intensities

    code.interact(local=locals())

    np.save(f"dataset/{flight_num}_{index}_alt.npy", altered_sample)
    
    


