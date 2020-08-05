import torch
import torch.nn as nn
from src.harmonization.pointnet import STNkd, PointNetfeat, Debug
import torch.nn.functional as F


class IntensityNet(nn.Module):
    def __init__(self, input_features, embed_dim, feature_transform, num_classes, neighborhood_size):
        super(IntensityNet, self).__init__()
        self.neighborhood_size = neighborhood_size
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True,
                feature_transform=feature_transform,
                num_features=input_features)

        self.fc1 = nn.Linear(1024+embed_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.0)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(),
        self.sigmoid = nn.Sigmoid()
        self.embed = nn.Embedding(50, embed_dim)
        self.debug = Debug()

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
        x, trans, trans_feat = self.feat(alt)
        x = torch.cat((x, fid_embed), dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.sigmoid(self.fc3(x))

        return x #, trans, trans_feat

        
