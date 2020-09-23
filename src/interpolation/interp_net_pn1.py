import torch
import torch.nn as nn
from src.models.pointnet.pointnet import STNkd, PointNetfeat
import torch.nn.functional as F
import code

class IntensityNetPN1(nn.Module):
    def __init__(self, neighborhood_size, input_features=8, embed_dim=3, num_classes=1):
        super(IntensityNetPN1, self).__init__()
        self.neighborhood_size = neighborhood_size
        self.feat = PointNetfeat(global_feat=True,
                feature_transform=False,
                num_features=input_features)

        self.fc_layer = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.Dropout(p=0.3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, num_classes)
                )

    def forward(self, batch):
        # Center the point cloud
        xyz = batch[:, :, :3]
        centered_xyz = xyz - xyz[:, 0, None]
        batch[:, :, :3] = centered_xyz

        if self.neighborhood_size == 0:
            alt = batch[:, 0, :]
            alt = alt.unsqueeze(1)
        else:
            alt = batch[:, 1:self.neighborhood_size+1, :]

        alt = alt[:, :, :8]
        alt = alt.transpose(1,2)
        x, trans, trans_feat = self.feat(alt)
        x = self.fc_layer(x)
        return F.sigmoid(x) #, trans, trans_feat
