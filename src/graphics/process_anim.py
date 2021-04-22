from manim import *
import numpy as np
from pathlib import Path
import code
from pptk import kdtree
from src.config.patch import patch_mp_connection_bpo_17560
from src.datasets.tools.overlap import get_hist_overlap, get_overlap_points

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
        patch_mp_connection_bpo_17560()

        # Build a test point cloud
        print("loading points")

        SCALE_FACTOR = 4  # expands the point clouds 

        pc1 = np.load("../datasets/dublin/data/npy/1.npy")
        pc2 = np.load("../datasets/dublin/data/npy/6.npy")
        pc12 = np.concatenate((pc1, pc2), axis=0)

        # Center
        pc1[:, :3] -= np.mean(pc12[:, :3], axis=0)
        pc2[:, :3] -= np.mean(pc12[:, :3], axis=0)
        pc12[:, :3] -= np.mean(pc12[:, :3], axis=0)



        # # Get overlap region
        if not (Path("aoi_idx.npy").exists() and Path("overlap_info.npy").exists()):
            print("generating overlap info and aoi_idx")
            config = {"workers": 8, 'tqdm': False}
            overlap_info, _ = get_hist_overlap(pc1, pc2, 10000)
            aoi_idx = get_overlap_points(pc1, overlap_info, config)
            np.save("aoi_idx", aoi_idx)
            np.save("overlap_info", overlap_info)

        else:
            print("Loading overlap info and aoi idx")
            aoi_idx = np.load("aoi_idx.npy")
            # overlap_info = np.load("overlap_info.npy", allow_pickle=True)

        # Normalize
        dist_max = np.sqrt(np.sum(((pc12[:, :3] - [0, 0, 0])**2), axis=1)).max()
        print(dist_max)
        pc1[:, :3] /= (dist_max/SCALE_FACTOR)
        pc2[:, :3] /= (dist_max/SCALE_FACTOR)

        print(f"Got overlap! Size: {aoi_idx.shape} ({aoi_idx.shape[0]/pc1.shape[0]})")
        # Get a reasonable sample of neighborhoods from pc1-->pc2
        kd = kdtree._build(pc2[:, :3], numprocs=8)
        query = kdtree._query(kd, pc1[aoi_idx, :3], k=10, dmax=1, numprocs=8)
        print("Query: ", len(query))

        neighborhoods = []
        targets = []
        sample_aoi = np.random.choice(len(query), size=100)
        
        for i in sample_aoi:
            neighborhoods.append(pc2[query[i]])

        print(len(neighborhoods))
        targets = pc1[aoi_idx][sample_aoi]
        print(targets.shape)

        pc1_group = PMobject().add_points(targets[:, :3], color=RED_E)

        pc2_group = []
        for i in range(len(neighborhoods)):            
            pc2_group.append(PMobject().add_points(
                neighborhoods[i][:, :3], color=BLUE_E))




        ### Visualize scans
        sample1 = np.random.choice(len(pc1), size=10000)
        sample2 = np.random.choice(len(pc2), size=10000)

        pc1_v = pc1[sample1]
        pc2_v = pc2[sample2]

        targets_group = PGroup(*pc1_group)
        neighborhood_group = PGroup(*pc2_group)

        # Setup Scene
        pm1 = PMobject()
        pm2 = PMobject()
        pm1.add_points(pc1_v[:, :3], color=BLUE_E)
        pm2.add_points(pc2_v[:, :3], color=RED_E)
        pm1.set_stroke_width(2)
        pm2.set_stroke_width(2)

        # See camera settings
        print("Starting Distance")
        print(self.camera.get_distance())
        print("Starting phi")
        print(self.camera.get_phi())
        print("Starting theta")
        print(self.camera.get_theta())


        # Create the scene
        self.add(pm1)
        self.add(pm2)
        self.add(targets_group)
        self.add(neighborhood_group)
        # self.add(overlap area)

        # Adjust camera some?
        self.move_camera(phi=0.10, theta=0.0)
        self.wait()

        self.play(
            FadeOutAndShift(pm1),
            FadeOutAndShift(pm2))
        
        self.move_camera(distance=10)
        self.wait()

        self.play(
            neighborhood_group.animate.shift([0, 0, .1]),
            targets_group.animate.shift([0, 0, 1]))

        self.wait()

        # self.remove(pm)
        self.wait(1)
