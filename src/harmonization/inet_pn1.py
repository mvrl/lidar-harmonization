import code
import torch
import torch.nn as nn
from src.models.pointnet.pointnet import STNkd, PointNetfeat
import torch.nn.functional as F


class IntensityNetPN1(nn.Module):
    def __init__(self, neighborhood_size, input_features=8, embed_dim=3, num_classes=1):
        super(IntensityNetPN1, self).__init__()
        self.neighborhood_size = neighborhood_size
        self.input_features = input_features
        self.camera_embed = nn.Embedding(45, embed_dim)
        self.feat = PointNetfeat(global_feat=True,
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
                nn.Linear(256, num_classes))

        self.harmonization = nn.Sequential(
                nn.Linear(num_classes+embed_dim+embed_dim, 8),
                nn.ReLU(),
                nn.Linear(8, num_classes))

        
        # Initializing harmonization weights to
        # identity speeds up convergence greatly
        self.harmonization[0].weight.data.copy_(torch.eye(8, num_classes+embed_dim))
        self.harmonization[2].weight.data.copy_(torch.eye(1, 8))


    def forward(self, batch):
        # [[neighborhood]] -> [H_p]
        # Neighborhood is the series of points closest to the first point.
        #    Point 0 (stripped) is the harmonization target  (in target scan)
        #    Point 1 is the interpolation target             (in target scan)
        #    Point 2 is closest neighbor                     (in source scan)
        #    Point 3-152 are the remaining closest neighbors (in source scan)

        # Center the point cloud
        xyz = batch[:, :, :3]
        centered_xyz = xyz - xyz[:, 0, None]
        batch[:, :, :3] = centered_xyz

        if self.neighborhood_size == 0:
            # use the interpolation target as the input (for testing only)
            alt = batch[:, 0, :]
            alt = alt.unsqueeze(1)
        else:
            # chop out interpolation target
            alt = batch[:, 1:self.neighborhood_size+1, :]
        
        # Camera embedding (for each neighborhood, the camera is the same for every point)
        source_camera = alt[:, 1, -1].long()
        target_camera = alt[:, 0, -1].long()
        
        source_camera_embed = self.camera_embed(source_camera)
        target_camera_embed = self.camera_embed(target_camera)
        camera_info = source_camera_embed - target_camera_embed
        alt = alt[:, :, :-1]  # remove camera data

        # Pointnet
        alt = alt.transpose(1, 2)
        x, trans, trans_feat = self.feat(alt)

        # interpolate value at center point
        interpolation = self.fc_layer(x)

        # fuse interpolation and camera info
        x = torch.cat((interpolation, camera_info), dim=1)

        # harmonize intensity to target camera
        harmonization = self.harmonization(x)

        return harmonization, interpolation


