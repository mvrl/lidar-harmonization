import json
import numpy as np

def apply_rf(rf_path, original, flight_num):
    """
    apply the response function to the data and return it as a new array
       original: nd.array point cloud with shape (N, C) where intensity is the fourth channel
       flight_num: index corresponding to the desired response function to be applied
    returns nd.array
    """

    # it seems ridiculous to do this every time
    
    with open(rf_path) as json_file:
        data = json.load(json_file)

    altered = original.copy()
    intensities = altered[:, 3]
    max_val = altered[:, 3].max()
    altered_intensities = np.interp(altered[:, 3],
                                    np.array(data['brightness'][str(flight_num)])*max_val,
                                    np.array(data['intensities'][str(flight_num)])*max_val)
    altered[:,3] = altered_intensities

    return altered
    

    

    
