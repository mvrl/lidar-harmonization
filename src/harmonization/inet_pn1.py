import code
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.harmonization.pointnet import STNkd, PointNetfeat
from scipy.interpolate import griddata


# it would be cool if interpolation was modular in this network, i.e., pointnet
#   or standard interpolation passed in as a parameter. 

class IntensityNet(nn.Module):
    def __init__(
            self,
            neighborhood_size, 
            interpolation_method="pointnet",
            input_features=8, 
            embed_dim=3, 
            num_classes=1, 
            h_hidden_size=100):

        super(IntensityNet, self).__init__()
        
        self.interpolation_method = interpolation_method
        self.neighborhood_size = neighborhood_size
        self.input_features = input_features
        self.camera_embed = nn.Embedding(45, embed_dim)
        
        if self.interpolation_method is "pointnet":
            self.feat = PointNetfeat(
                global_feat=True,
                feature_transform=True,
                num_features=self.input_features)

            self.fc_layer = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.Dropout(p=0.0),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 1))

        if self.interpolation_method not in ["pointnet", "linear", "nearest", "cubic"]:
            raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")

        self.harmonization = nn.Sequential(
            nn.Linear(1+embed_dim, h_hidden_size),
            nn.Dropout(p=0.0),        
            nn.ReLU(),
            nn.Linear(h_hidden_size, num_classes))


        # Initializing to identify to speed up convergence
        self.harmonization[0].weight.data.copy_(
            torch.eye(
                h_hidden_size, 
                1+embed_dim))

        self.harmonization[3].weight.data.copy_(
            torch.eye(
                num_classes, 
                h_hidden_size))


    def forward(self, batch):
        # [[neighborhood]] -> [H_p]
        # Neighborhood is the series of points closest to the first point.
        #    Point -1 (stripped) is the harmonization target  (target scan)
        #    Point 0 is the interpolation target              (target scan)
        #    Point 1 is closest neighbor                      (source scan)
        #    Point 3-151 are the remaining closest neighbors  (source scan)

        with torch.no_grad():
            # Center the point cloud
            xyz = batch[:, :, :3]
            batch[:, :, :3] = xyz - xyz[:, 0, None]

            # Camera embedding is in the last channel
            target_camera = batch[:, 0, -1].long()
            source_camera = batch[:, 1, -1].long()

            # Loss calculation requires that in overlap examples are compared 
            #     against target scan while out of overlap samples are compared
            #     against the interpolation target. This line determines which
            #     examples are source-target examples and which are source-source
            #     (i.e., which required interpolation and which did not). 
            #     Source-source examples will not be compared against the target
            #     camera's ground truth value. 
            ss = torch.where(~((target_camera - source_camera).bool()) == True)

            batch = batch[:, :, :-1]  # remove camera data
            if self.neighborhood_size == 0:
                # use the interpolation target as the input (for testing only)
                batch = batch[:, 1, :]
                batch = batch.unsqueeze(1)
            else:
                # chop out harmonization and interpolation targets
                batch = batch[:, 2:self.neighborhood_size+2, :]
        
        if self.interpolation_method is "pointnet":
            batch = batch.transpose(1, 2)
            x, trans, trans_feat = self.feat(batch)

            # interpolate value at center point
            interpolation = self.fc_layer(x)

        else:
            # yucky implementation since we rely on numpy and scipy.
            # improved one day? See https://github.com/pytorch/pytorch/issues/50341
            # this doesn't work for the ndim=0 test case
            with torch.no_grad():
                device = batch.device
                if str(device) is not 'cpu':
                    batch = batch.cpu()

                interpolation = torch.cat([
                    torch.tensor(griddata(
                        n[1:self.neighborhood_size+1, :3],
                        n[1:self.neighborhood_size+1, 3],
                        [0, 0, 0], method=self.interpolation_method
                        )) for n in batch]
                    )

                interpolation = interpolation.unsqueeze(1).to(device)

        source_camera_embed = self.camera_embed(source_camera)
        target_camera_embed = self.camera_embed(target_camera)
        camera_info = source_camera_embed - target_camera_embed

        # fuse interpolation and camera info
        x = torch.cat((interpolation, camera_info), dim=1)

        # harmonize intensity to target camera
        harmonization = self.harmonization(x)

        return harmonization, interpolation, ss


