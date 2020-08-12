import torch
import torch.nn as nn
from src.harmonization.pointnet.pointnet import STNkd, PointNetfeat, PointNetMLP
from src.harmonization.pointnet2.pointnet2_ssg_cls import PointNet2ClassificationSSG
import torch.nn.functional as F
from functools import partial
import code


def forward_pn1(pointcloud, fid_embed, feat=None, fc_layer=None):
    pointcloud = pointcloud.transpose(1, 2)

    x, trans, trans_feat = feat(pointcloud)
    x = torch.cat((x, fid_embed), dim=1)
    x = fc_layer(features.double())
    return x

def forward_pn2(pointcloud, fid_embed, pointnet2=None, fc_layer=None):
    xyz, features = pointnet2(pointcloud)
    features = torch.cat((features.squeeze(-1), fid_embed.float()), dim=1)
    x = self.fc_layer(features.double())
    return x

class Model:
    def __init__(self, model_type, params):
        self.model_type = model_type
        self.params = params
        if self.model_type is "pointnet1":
            self.forward = partial(
                    forward_pn1, 
                    feat=params["feat"], 
                    fc_layer=params["fc_layer"])

        if self.model_type is "pointnet2":
            self.forward = partial(
                    forward_pn2,
                    pointnet2=params["pointnet2"],
                    fc_layer=params["fc_layer"])

        if self.model_type is "simple_mlp":
            pass

    def __call__(self, pointcloud, fid_embed):
        return self.forward(pointcloud, fid_embed)


class IntensityNet(nn.Module):
    def __init__(self, neighborhood_size, model_name="pointnet1", input_features=8, embed_dim=3, output_size=1):
        super(IntensityNet, self).__init__()
        self.neighborhood_size = neighborhood_size
        self.model_name = model_name
        self.params = {}

        if self.model_name is "pointnet1":
            self.params["feat"] = PointNetfeat(
                    global_feat=True,
                    feature_transform=True,
                    num_features=input_features)
            self.params["fc_layer"] = PointNetMLP(embed_dim, output_size)
        
        if self.model_name is "pointnet2":
            self.params["pointnet2"] = PointNet2ClassificationSSG()
            self.params["fc_layer"] = PointNetMLP(embed_dim, output_size)

        if self.model_name is "simple_mlp":
            pass

        self.model = Model(self.model_name, self.params)
        
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
        fid_embed = self.embed(fid)
 
        alt = alt[:, :, :8].float()

        x = self.model(alt, fid_embed)

        return F.sigmoid(x)

        
