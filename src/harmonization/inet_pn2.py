import torch
import torch.nn as nn
from src.harmonization.pointnet2.pointnet2_ssg_cls import PointNet2ClassificationSSG
import torch.nn.functional as F
import code


class IntensityNetPN2(nn.Module):
    def __init__(self, neighborhood_size, input_features=8, embed_dim=3, num_classes=1):
        super(IntensityNetPN2, self).__init__()
        self.neighborhood_size = neighborhood_size
        
        self.pointnet2 = PointNet2ClassificationSSG()
        self.fc_layer = nn.Sequential(
                nn.Linear(1024+embed_dim, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                nn.Linear(512, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes),
                ).float()
        
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

        # alt = alt.transpose(1, 2)  # for pointnet1
        fid_embed = self.embed(fid)
        alt = alt.float()  # for pointnet2
        xyz, features = self.pointnet2(alt)
        features = torch.cat((features.squeeze(-1), fid_embed.float()), dim=1)
        x = self.fc_layer(features.double())
        

        return F.sigmoid(x)
