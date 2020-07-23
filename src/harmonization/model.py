import code
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.optim import Adam

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from src.dataset.tools.lidar_dataset import LidarDataset
from src.dataset.tools.transforms import LoadNP, CloudCenter, CloudIntensityNormalize, CloudRotateX, CloudRotateY, CloudRotateZ, CloudJitter, GetTargets

from src.dataset.tools.metrics import create_kde, create_bar
from src.harmonization.pointnet import PointNetfeat, STNkd, Debug
from src.ex_pl.extended_lightning import ExtendedLightningModule        
        
class IntensityNet(ExtendedLightningModule):
    def __init__(self,
                 train_dataset_csv,
                 val_dataset_csv,
                 test_dataset_csv,
                 qual_dataset_csv,
                 neighborhood_size=0,  # set N=0 for single flight test cases
                 num_classes=1,
                 batch_size=50,
                 num_workers=8,
                 embed_dim=3,
                 input_features=8,
                 dual_flight=None,
                 feature_transform=False,
                 results_dir=r"results/"):

        super(IntensityNet, self).__init__()

        # Configuration Options
        self.neighborhood_size = neighborhood_size
        self.batch_size = batch_size
        self.num_workers = num_workers 
        self.dual_flight = dual_flight 

        # prepare results directory
        self.results_root = Path(results_dir)
        self.results_dir = self.results_root /  f"{self.neighborhood_size}"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Dataset Definitions
        self.train_dataset_csv = train_dataset_csv
        self.val_dataset_csv = val_dataset_csv
        self.test_dataset_csv = test_dataset_csv
        self.qual_dataset_csv = qual_dataset_csv

        # Network Layers
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
        self.relu = nn.ReLU()
        
        self.embed = nn.Embedding(50, embed_dim)
        self.debug = Debug()

    def forward(self, batch):

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
        x = self.fc3(x)
        
        return x #, trans, trans_feat

    def prepare_data(self):
        self.criterion = nn.SmoothL1Loss()
        transformations = Compose([
            LoadNP(),
            CloudCenter(),
            CloudIntensityNormalize(512),
            GetTargets()])

        self.train_dataset = LidarDataset(
                self.train_dataset_csv,
                transform=transformations,
                dual_flight=self.dual_flight)

        self.val_dataset = LidarDataset(
                self.val_dataset_csv,
                transform=transformations,
                dual_flight=self.dual_flight)

        self.test_dataset = LidarDataset(
                self.test_dataset_csv,
                transform=transformations,
                dual_flight=self.dual_flight)

        self.qual_dataset = LidarDataset(
                self.qual_dataset_csv,
                transform=transformations,
                dual_flight=self.dual_flight)


    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                drop_last=True)


    def val_dataloader(self):
        return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True)
   
    def test_dataloader(self):
        return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True)

    def qual_dataloader(self):
        return DataLoader(
                self.qual_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                drop_last=True)

    def training_step(self, batch, batch_idx):
        data, target = batch
        alt = data[:, 0, 3]
        my_fid = int(data[0, 0, 8])

        output = self.forward(data)
        loss = self.criterion(output.squeeze(), target)
        return {'loss': loss, 
                'metrics': {'target': target, 'output': output.squeeze()}}


    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data.double())
        loss = self.criterion(output.squeeze(), target.double())

        return {'val_loss': loss, 
                'metrics': {'target':target, 'output':output.squeeze()}}   


    def test_step(self, batch, batch_idx):
        data, target = batch
        alt = data[:, 0, 3]
        output = self.forward(data.double())

        return {'metrics': {'target': target, 'output': output.squeeze()}}

    def qual_step(self, batch, batch_idx):
        data, target = batch
        alt = data[: 0, 3]
        output = self.forward(data.double())

        return {'metrics': {'target': target, 'output': output.squeeze()}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.targets = torch.stack([x['metrics']['target'] for x in outputs])
        self.predictions = torch.stack([x['metrics']['output'] for x in outputs])
        
        self.mae = torch.mean(torch.abs(self.targets.flatten() - self.predictions.flatten())).item()
        
        return {'avg_train_loss': avg_loss}
        

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.targets = torch.stack([x['metrics']['target'] for x in outputs])
        self.predictions = torch.stack([x['metrics']['output'] for x in outputs])
        
        self.mae = torch.mean(torch.abs(self.targets.flatten() - self.predictions.flatten())).item()

        return {'avg_val_loss': avg_loss}

    def test_epoch_end(self, outputs):
        self.targets = torch.stack([x['metrics']['target'] for x in outputs])
        self.predictions = torch.stack([x['metrics']['output'] for x in outputs])

        self.mae = torch.mean(torch.abs(self.targets.flatten() - self.predictions.flatten())).item()


    def qual_epoch_end(self, outputs):
        self.targets = torch.stack([x['metrics']['target'] for x in outputs])
        self.predictions = torch.stack([x['metrics']['output'] for x in outputs])

        self.mae = torch.mean(torch.abs(self.targets.flatten() - self.predictions.flatten())).item()
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters())
        # scheduler = CyclicLR(
        #       optimizer,
        #       1e-6,
        #       1e-3,
        #       step_size_up=len(self.train_dataset)//self.batch_size//2,
        #       scale_fn = lambda x: 1 / ((5/4.) ** (x-1)),
        #       cycle_momentum=False)

        scheduler = StepLR(optimizer, step_size=1)
        return [optimizer], [scheduler]


