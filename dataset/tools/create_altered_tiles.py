import time
import code
import numpy as np
import json
from pathlib import Path

def random_mapping():
    # there are 41 flights, 160 response functions
    mapping = np.random.choice(160, size=41)
    return mapping
    

def create_altered_tiles(path, mapping=None):

    start_time = time.time()

    # Read in the response functions file
    with open('response_functions.json') as json_file:
        data = json.load(json_file)

    print(data.keys())  # names scales brightness intensities

    # get dataset file directory ready
    dataset_path = Path(path)
    gt_path = dataset_path / "gt"
    alt_path = dataset_path / "alt"
    alt_path.mkdir(exist_ok=True)

    files = [f for f in gt_path.glob('*.npy')]
    file_count = len(files)

    print("found %s files to alter" % file_count)

    for file_path in files:
        file_name = file_path.stem
        flight_num, index = file_name.split('_')[0], file_name.split('_')[1]
        
        curr_sample = np.load(file_path)
        curr_intensities = curr_sample[:,3]

        # 65534.0 is max value for intensity across flights
        altered_intensities = np.interp(
            curr_intensities,
            np.array(data['brightness'][str(mapping[int(flight_num)])])*512,
            np.array(data['intensities'][str(mapping[int(flight_num)])])*512,
        )

        altered_sample = np.copy(curr_sample)
        altered_sample[:,3] = altered_intensities

        np.save(alt_path / f"{flight_num}_{index}.npy", altered_sample)

    print(f"finished in {time.time() - start_time} seconds")
    
    
if __name__=='__main__':
    mapping = random_mapping()
    create_altered_tiles('synth')
