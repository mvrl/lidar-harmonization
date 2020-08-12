import torch
import torch.nn as nn
from src.harmonization.pointnet import STNkd, PointNetfeat, Debug
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """ 
    Ablation study on features
    """
    def __init__(self, neighborhood_size=0, embed_dim=3, num_classes=1):
        super(IntensityNet, self).__init__()
        self.neighborhood_size = neighborhood_size
        self.mlp = nn.Sequential(
                nn.Linear(8+emebd_dim, 6),
                nn.ReLU(True),
                nn.Linear(6, 4),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Linear(4, 2),
                nn.ReLU(True),
                nn.Linear(2, 1))

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

        alt = alt.transpose(1, 2)

        fid_embed = self.embed(fid)

        x = torch.cat((x, fid_embed), dim=1)
        x = self.mlp(x)
        x = self.sigmoid(self.fc3(x))

        return x #, trans, trans_feat
