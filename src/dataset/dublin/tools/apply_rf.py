import json
import numpy as np
import code


class ApplyResponseFunction:
    def __init__(self, rf_path, mapping, max_intensity):
        self.mapping = np.load(mapping)
        self.rf_path = rf_path
        self.max_intensity = max_intensity
        with open(self.rf_path) as json_file:
            self.rf_data = json.load(json_file)

    def __call__(self, original, fid, noise=False):
        altered = original.copy()
        intensities = altered[:, 3]

        # pull the correct mapping from file
        mapped_trans = str(self.mapping[fid])
                
        xp = np.fromstring(self.rf_data[mapped_trans]['B'], sep=' ')
        fp = np.fromstring(self.rf_data[mapped_trans]['I'], sep=' ')
        
        if noise:
            xp+=np.random.normal(0, .1, xp.shape)
            fp+=np.random.normal(0, .1, fp.shape)

        # apply the transformation
        altered_intensities = np.interp(
            altered[:, 3],
            xp*self.max_intensity,
            fp*self.max_intensity)

        altered[:,3] = altered_intensities

        return altered
    