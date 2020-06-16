import json
import numpy as np
import code

def apply_rf(rf_path, original, flight_num, max_val, rf_data=None):
    """
    apply the response function to the data and return it as a new array
       original: nd.array point cloud with shape (N, C) where intensity is the fourth channel
       flight_num: index corresponding to the desired response function to be applied
    returns nd.array
    """

    if not rf_data:
        with open(rf_path) as json_file:
            data = json.load(json_file)

    else:
        data = rf_data
    altered = original.copy()
    intensities = altered[:, 3]

    altered_intensities = np.interp(
            altered[:, 3],
            np.fromstring(data[str(flight_num)]['B'], sep=' ')*max_val,
            np.fromstring(data[str(flight_num)]['I'], sep=' ')*max_val)

    altered[:,3] = altered_intensities

    return altered
    

    

    
