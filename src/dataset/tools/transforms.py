import numpy as np
import torch

class LoadNP(object):
    def __call__(self, example):
        # Load the numpy files
        return np.loadtxt(example)


class CloudCenter(object):
    # currenty being handled in forward pass
    def __call__(self, example):
        # Center the neighborhood
        ex = example.copy()
        ex[:,:3] -= ex[0, :3]
        return ex

    
class CloudIntensityNormalize(object):
    def __init__(self, max):
        self.max = max
    def __call__(self, example):
        # Normalize the intensity values
        example[:, 3]/=self.max
        return example

class CloudAngleNormalize(object):
    # values go from -90 to 90
    # transform this to -1 to 1
    def __call__(self, example):
        ex = example.copy()
        ex[:, 4]/=90.0
        return ex

class GetTargets(object):
    # return interpolation and harmonization values
    def __call__(self, example):
        ex = example.copy()
        target_i = ex[0, 3]
        source_i = ex[1, 3]

        # the center point must stay in the model so that
        # it can be centered, it will be removed in forward pass
        ex = ex[1:, :]
        return (ex, target_i, source_i)


class ToTensor(object):
    def __call__(self, example):
        return torch.from_numpy(example)
                
class CloudRotateX(object):
    def __init__(self):
        pass

    def __call__(self, example):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cosval, -sinval],
            [0, sinval, cosval]])

        example[:, :3] = np.dot(example[:, :3], rotation_matrix)
        return example
    
class CloudRotateY(object):
    def __init__(self):
        pass

    def __call__(self, example):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([
                [cosval, 0, sinval],
                [0, 1, 0],
                [-sinval, 0, cosval]])

        example[:, :3] = np.dot(example[:, :3], rotation_matrix)

        return example


class CloudRotateZ(object):
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    def __init__(self):
        pass

    def __call__(self, example):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        example[:,:3] = np.dot(example[:,:3], rotation_matrix)
        
        return example

class CloudJitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, example):
        jittered_data = np.clip(
            self.sigma*np.random.randn(
                 example.shape[0], 3),
            -1*self.clip,
            self.clip)
        
        example[:,:3] += jittered_data
                
        return example

    

