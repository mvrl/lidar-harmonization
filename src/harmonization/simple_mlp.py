import code
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """ 
    Ablation study on features
    """
    def __init__(self, neighborhood_size=0, embed_dim=3, num_features=4, num_classes=1, use_surface_characteristics=True, use_scan_angle=True):
        super(SimpleMLP, self).__init__()
        self.neighborhood_size = neighborhood_size  # neighborhood_size
        self.num_features = 5

        self.mlp = nn.Sequential(
                nn.Linear(self.num_features+embed_dim, 10),
                nn.ReLU(True),
                nn.Linear(10, 1),
                )
        
        self.embed = nn.Embedding(50, embed_dim)
        self.sigmoid = nn.Sigmoid()

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
        # chop out unwanted features (scan angle, surface normals)
        # alt = alt[:, :, 3] # literally just look at intensity and nothing else
        # alt = alt[:, :, 3:5] # look at intensity, scan angle
        alt = alt[:, :, 3:8] # look at intensity, scan angle, normals
        fid_embed = self.embed(fid)
        x = torch.cat((alt.squeeze(), fid_embed), dim=1)
        x = self.mlp(x)

        return x #, trans, trans_feat
