from manim import *
import numpy as np
import code

from src.datasets.tools.overlap import 

# Animation story board:
# 1: show two scans (different colors, not harmonized!)
# 2: Show our method
#    a. Isolate the overlap region
#    b. Split into neighborhoods (???)
#    c. Examine a single neighborhood
#    d. Pointnet extracts local features from source, interpolates point at target
#    e. Learn a mapping from source --> target
#    f. Zoom out, apply new color to source scan

class ProcessAnim(ThreeDScene):
    def construct(self):

    	# Build a test point cloud
    	print("loading points")

    	pc1 = np.load("../datasets/dublin/data/npy/1.npy")
    	pc2 = np.load("../datasets/dublin/data/npy/6.npy")

    	# Get overlap region
    	config = {"workers": 8, 'tqdm': False}
    	overlap_info, _ = get_hist_overlap(pc1, pc2, sample_overla)
    	aoi_idx = get_overlap_points(pc1, overlap_info, config)

    	### Visualize scans
    	sample1 = np.random.choice(len(pc1), size=10000)
    	sample2 = np.random.choice(len(pc2), size=10000)

    	pc1_v = pc1[sample1]
    	pc2_v = pc2[sample2]
    	pc12 = np.concatenate((pc1, pc2), axis=0)
    	    	
    	# Center
    	pc1_v[:, :3] -= np.mean(pc12[:, :3], axis=0)
    	pc2_v[:, :3] -= np.mean(pc12[:, :3], axis=0)
    	pc12[:, :3] -= np.mean(pc12[:, :3], axis=0)
    	
    	# Normalize
    	dist_max = np.sqrt(np.sum(((pc12[:, :3] - [0, 0, 0])**2), axis=1)).max()
    	print(dist_max)
    	pc1_v[:, :3] /= (dist_max/4)
    	pc2_v[:, :3] /= (dist_max/4)

    	# Setup Scene
    	pm = PMobject()
    	pm.add_points(pc1_v[:, :3], color=BLUE_E)
    	pm.add_points(pc2_v[:, :3], color=RED_E)
    	pm.set_stroke_width(2)

    	# See camera settings
    	print("Starting Distance")
    	print(self.camera.get_distance())
    	print("Starting phi")
    	print(self.camera.get_phi())
    	print("Starting theta")
    	print(self.camera.get_theta())


    	# Create the scene and perform animations
    	self.add(pm)
    	self.move_camera(phi=0.10, theta=0.0)
    	self.wait(5)
