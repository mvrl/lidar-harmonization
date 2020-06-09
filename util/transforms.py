import numpy as np
import torch

class LoadNP(object):
    def __call__(self, sample):
        # Load the numpy files
        gt, alt = sample
        return (np.load(gt), np.load(alt))


class CloudCenter(object):
    def __call__(self, sample):
        # Center the neighborhood
        gt, alt = sample
        centered_gt = np.copy(gt)
        centered_gt[:,:3] = gt[:,:3] - gt[0][:3]
        centered_alt = np.copy(alt)
        centered_alt[:,:3] = alt[:,:3] - alt[0][:3]
        return (centered_gt, centered_alt)

    
class CloudIntensityNormalize(object):
    def __init__(self, max):
        self.max = max
    def __call__(self, sample):
        # Normalize the intensity values
        gt, alt = sample
        gt[:, 3]/=self.max
        alt[:, 3]/=self.max
        return (gt, alt)

        
class ToTensor(object):
    def __call__(self, sample):
        gt, alt = sample
        return (torch.from_numpy(gt),
                torch.from_numpy(alt))

class ToTensorBD(object):
    def __call__(self, sample):
        cloud, flight_ids = sample
        return (torch.from_numpy(cloud), torch.tensor(flight_ids))

    
class CloudNormalizeBD(object):
    def __call__(self, sample):
        cloud, flight_id = sample
        centered_cloud = np.copy(cloud)
        centered_cloud[:,:3] = cloud[:,:3] - cloud[0][:3]
        return (centered_cloud, flight_id)

    
class CloudAugment(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        gt, alt = sample
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        gt[:,:3] = np.dot(gt[:,:3], rotation_matrix)
        alt[:,:3] = np.dot(alt[:,:3], rotation_matrix)
        
        return (gt, alt)


# not implemented below this point 
class CloudJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, sample):
        gt, alt, num, path = sample
        jittered_data = np.clip(
            self.sigma*np.random.randn(
                 gt.shape[0], 3),
            -1*self.clip,
            self.clip)
        
        cloud[:3,:] = jittered_data + cloud[:3,:]
        return cloud

    

