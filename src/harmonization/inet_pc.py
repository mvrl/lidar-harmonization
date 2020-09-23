import torch
import torch.nn as nn
from src.models.pointconv.pointconv_util import PointConvDensitySetAbstraction
import torch.nn.functional as F
import code


class IntensityNetPC(nn.Module):
    def __init__(self, neighborhood_size, input_features=8, embed_dim=3, num_classes=1):
        super(IntensityNetPC, self).__init__()
        self.neighborhood_size = neighborhood_size
        
        self.sa1 = PointConvDensitySetAbstraction(
                 npoint=neighborhood_size, 
                 nsample=32, 
                 in_channel=3, 
                 mlp=[64, 64, 128], 
                 bandwidth = 0.1, 
                 group_all=False)

        self.sa2 = PointConvDensitySetAbstraction(
                npoint=128, 
                nsample=64, 
                in_channel=128 + 3, 
                mlp=[128, 128, 256], 
                bandwidth = 0.2, 
                group_all=False)

        self.sa3 = PointConvDensitySetAbstraction(
                npoint=1, 
                nsample=None, 
                in_channel=256 + 3, 
                mlp=[256, 512, 1024], 
                bandwidth = 0.4, 
                group_all=True)

        self.fc_layer = nn.Sequential(
                nn.Linear(1024+embed_dim, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.Dropout(0.4),
                nn.Linear(512, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.4),
                nn.Linear(256, num_classes),
                )
        
        self.embed = nn.Embedding(50, embed_dim)

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

        fid = alt[:, :, 8][:, 0].long()
        alt = alt[:, :, :8]
        # alt = alt.transpose(1, 2)

        fid_embed = self.embed(fid)
        xyz, features = alt[:, :, :3], alt[:, :, 3:]
        code.interact(local=locals())
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        code.interact(local=locals())
        x = l3_points.view(batch.shape[0], 1024)
        x = torch.cat((x, fid_embed.float()), dim=1)
        x = self.fc_layer(features.double())
        
        return F.sigmoid(x)
