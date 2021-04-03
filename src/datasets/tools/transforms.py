import numpy as np
import torch
from src.datasets.dublin.tools.apply_rf import ApplyResponseFunction
from src.datasets.dublin.tools.shift import sigmoid

class LoadNP(object):
    def __call__(self, example):
        # Load the numpy files
        return np.loadtxt(example)

class CloudAngleNormalize(object):
    # values go from -90 to 90
    # transform this to -1 to 1
    def __call__(self, example):
        ex = example.copy()
        ex[:, 4]/=90.0
        return ex

class Corruption(object):
    # for dublin only
    def __init__(self, **kwargs):
        dorf = kwargs['dorf_path']
        mapping = kwargs['mapping_path'] 
        self.ARF = ApplyResponseFunction(dorf, mapping)

    def __call__(self, example):
        # we need to save a copy of the ground truth point
        ex = example.copy()
        gt_copy = example[0, :].copy()
        gt_copy = np.expand_dims(gt_copy, 0)

        # find the point source for the neighborhood
        fid = int(example[-1, 8])

        # apply the pre-assigned corruption
        alt = self.ARF(ex.copy(), fid)
        alt[:, 8] = fid
        ex = np.concatenate((gt_copy, alt), axis=0)

        return ex

class GlobalShift(object):
    # for dublin only
    def __init__(self, **params):
        self.bounds = np.load(params['bounds_path'])
        self.min_x = self.bounds[0][0]
        self.max_x = self.bounds[0][1]

        self.floor = float(params['sig_floor'])
        self.center = float(params['sig_center'])
        self.l = int(params['sig_l'])
        self.s = float(params['sig_s'])
    
    def __call__(self, example):
        # global shift affects the ground truth (idx=0)
        ex = example.copy()
        x = ex[:, 0].copy()
        x = (x - self.min_x)/(self.max_x - self.min_x)
        ex[:, 3] *= sigmoid(x, 
                            h=self.center, 
                            v=self.floor, 
                            l=self.l, 
                            s=self.s)

        return ex

                
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
