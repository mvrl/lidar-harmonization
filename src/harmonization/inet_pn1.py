import code
import torch
import torch.nn as nn
from src.models.pointnet.pointnet import STNkd, PointNetfeat
import torch.nn.functional as F


class IntensityNetPN1(nn.Module):
    def __init__(self, neighborhood_size, input_features=8, embed_dim=3, num_classes=1, h_hidden_size=100):
        super(IntensityNetPN1, self).__init__()
        self.neighborhood_size = neighborhood_size
        self.input_features = input_features
        self.camera_embed = nn.Embedding(45, embed_dim)
        self.feat = PointNetfeat(
            global_feat=True,
            feature_transform=False,
            num_features=self.input_features)

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1))

        self.harmonization = nn.Sequential(
            nn.Linear(1+embed_dim, h_hidden_size),
            nn.Dropout(p=0.3),        
            nn.ReLU(),
            nn.Linear(h_hidden_size, num_classes))

        
        # Initializing harmonization weights to
        # identity speeds up convergence greatly
        self.harmonization[0].weight.data.copy_(
            torch.eye(
                h_hidden_size, 
                1+embed_dim))
        self.harmonization[3].weight.data.copy_(
            torch.eye(
                num_classes, 
                h_hidden_size))


    def forward(self, batch):
        # [[neighborhood]] -> [H_p]
        # Neighborhood is the series of points closest to the first point.
        #    Point -1 (stripped) is the harmonization target  (target scan)
        #    Point 0 is the interpolation target              (target scan)
        #    Point 1 is closest neighbor                      (source scan)
        #    Point 3-151 are the remaining closest neighbors  (source scan)

        # Center the point cloud
        xyz = batch[:, :, :3]
        batch[:, :, :3] = xyz - xyz[:, 0, None]
        

        # Camera embedding is in the last channel
        target_camera = batch[:, 0, -1].long()
        source_camera = batch[:, 1, -1].long()

        # Loss calculation requires that in overlap examples are compared 
        #     against target scan while out of overlap samples are compared
        #     against the interpolation target. This line determines which
        #     examples are source-target examples and which are source-source
        #     (i.e., which required interpolation and which did not). 
        #     Source-source examples will not be compared against the target
        #     camera's ground truth value. 
        ss = torch.where(~((target_camera - source_camera).bool()) == True)
        
        # code.interact(local=locals())
        
        source_camera_embed = self.camera_embed(source_camera)
        target_camera_embed = self.camera_embed(target_camera)
        camera_info = source_camera_embed - target_camera_embed

        batch = batch[:, :, :-1]  # remove camera data

        if self.neighborhood_size == 0:
            # use the interpolation target as the input (for testing only)
            batch = batch[:, 0, :]
            batch = batch.unsqueeze(1)
        else:
            # chop out interpolation target
            batch = batch[:, 1:self.neighborhood_size+1, :]
        
        # Pointnet
        batch = batch.transpose(1, 2)
        x, trans, trans_feat = self.feat(batch)

        # interpolate value at center point
        interpolation = self.fc_layer(x)

        # fuse interpolation and camera info
        x = torch.cat((interpolation, camera_info), dim=1)

        # harmonize intensity to target camera
        harmonization = self.harmonization(x)

        return harmonization, interpolation, ss


