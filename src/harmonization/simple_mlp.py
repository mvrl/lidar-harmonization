import code
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """ 
    Ablation study on features
    """
    def __init__(self, neighborhood_size=0, embed_dim=3, num_features, num_classes=1):
        super(SimpleMLP, self).__init__()
        self.neighborhood_size = 0  # neighborhood_size
        self.mlp = nn.Sequential(
                nn.Linear(8+embed_dim, 6),
                nn.ReLU(True),
                nn.Linear(6, 4),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Linear(4, 2),
                nn.ReLU(True),
                nn.Linear(2, 1))
        
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
        alt = alt.squeeze()

        # chop out unwanted features (scan angle, surface normals)
        alt = alt[:, :num_features, :]
        fid_embed = self.embed(fid)
        x = torch.cat((alt, fid_embed), dim=1)
        x = self.mlp(x)
        x = self.sigmoid(x)

        return x #, trans, trans_feat
