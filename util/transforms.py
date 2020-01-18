import numpy as np
import torch


class LoadNP(object):
    def __call__(self, sample):
        # Load the numpy files
        gt, alt, num, path = sample
        return (np.load(gt), np.load(alt), num, path)


class CloudNormalize(object):
    def __init__(self):
        pass
    
    def __call__(self, sample):
        # Center the neighborhood
        gt, alt, num, path = sample
        centered_gt = np.copy(gt)
        centered_gt[:,:3] = gt[:,:3] - gt[0][:3]
        centered_alt = np.copy(alt)
        centered_alt[:,:3] = alt[:,:3] - alt[0][:3]
        return (centered_gt, centered_alt, num, path)

class ToTensor(object):
    def __call__(self, sample):
        gt, alt, num, path = sample
        return (torch.from_numpy(gt), torch.from_numpy(alt), num, path)


""" not implemented below this point """
class CloudAugment(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        cloud = sample
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        cloud[:,:3] = np.dot(cloud[:,:3], rotation_matrix)
        return cloud

    
class CloudJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, sample):
        cloud = sample
        jittered_data = np.clip(
            self.sigma*np.random.randn(
                3, cloud.shape[1]),
            -1*self.clip,
            self.clip)
        
        cloud[:3,:] = jittered_data + cloud[:3,:]
        return cloud

    

