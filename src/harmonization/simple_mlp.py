import code
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """ 
    Ablation study on features
    """
    def __init__(self, neighborhood_size=0, embed_dim=3, num_features=4, num_classes=1):
        super(SimpleMLP, self).__init__()
        self.neighborhood_size = 0  # neighborhood_size
        self.num_features = num_features
        self.mlp = nn.Sequential(
                nn.Linear(1, 1),
                nn.ReLU(True),
                nn.Linear(1, 1),
                # nn.ReLU(True),
                # nn.Linear(4, 1)
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
        alt = alt[:, :, 4].unsqueeze(-1)  # literally just look at intensity and nothing else
        
        # fid_embed = self.embed(fid)
        # x = torch.cat((alt, fid_embed), dim=1)
        x = self.mlp(alt)
        x = self.sigmoid(x)

        return x #, trans, trans_feat
