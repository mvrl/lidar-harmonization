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
from src.dataset.tools.transforms import LoadNP, CloudCenter, CloudIntensityNormalize, CloudRotateX, CloudRotateY, CloudRotateZ, CloudJitter, GetTargets, GetTargetsInterp, CloudAngleNormalize

from src.dataset.tools.metrics import create_kde, create_bar
from src.interpolation.interp_net_pn1 import IntensityNetPN1
from src.interpolation.interp_net_pn2 import IntensityNetPN2

from src.ex_pl.extended_lightning import ExtendedLightningModule        
        
class InterpolationNet(ExtendedLightningModule):
    def __init__(self,
                 train_dataset_csv,
                 val_dataset_csv,
                 test_dataset_csv,
                 qual_dataset_csv,
                 model_name="pointnet1",
                 neighborhood_size=0,  # set N=0 for single flight test cases
                 batch_size=50,
                 num_workers=1,
                 dual_flight=None,
                 feature_transform=False,
                 results_dir=r"results/"):

        super(InterpolationNet, self).__init__()

        # Configuration Options
        self.neighborhood_size = neighborhood_size
        self.batch_size = batch_size
        self.num_workers = num_workers 
        self.dual_flight = dual_flight 

        # Dataset Definitions
        self.train_dataset_csv = Path(train_dataset_csv)
        self.val_dataset_csv = Path(val_dataset_csv)
        self.test_dataset_csv = Path(test_dataset_csv)
        self.qual_dataset_csv = Path(qual_dataset_csv)
        
        # prepare results directory
        self.results_root = Path(results_dir)
        self.results_dir = self.results_root /  f"{self.neighborhood_size}_df" if self.dual_flight else (
                self.results_root / f"{self.neighborhood_size}")

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.qual_in = True if "_in_" in str(self.qual_dataset_csv) else False
        

        # Misc:
        self.xyzi = None

        # Network
        if model_name is "pointnet1":
            self.net = IntensityNetPN1(self.neighborhood_size).float()
        if model_name is "pointnet2":
            self.net = IntensityNetPN2(self.neighborhood_size).float()
        if model_name is "simple_mlp":
            self.net = SimpleMLP(self.neighborhood_size).float()
        if model_name is "pointconv":
            self.net = IntensityNetPC(self.neighborhood_size).float()

    def forward(self, batch):
        return self.net(batch)

    def prepare_data(self):
        self.criterion = nn.SmoothL1Loss()
        transformations = Compose([
            LoadNP(),
            CloudIntensityNormalize(512),
            CloudAngleNormalize(),
            GetTargetsInterp()])

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
                # pin_memory=True)


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
        output = self.forward(data)
        loss = self.criterion(output.squeeze(), target)
        return {'loss': loss, 
                'metrics': {'target': target, 'output': output.squeeze()}}


    def validation_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data)
        loss = self.criterion(output.squeeze(), target)

        return {'val_loss': loss, 
                'metrics': {'target':target, 'output':output.squeeze()}}   


    def test_step(self, batch, batch_idx):
        data, target = batch
        alt = data[:, 0, 3]
        output = self.forward(data)
        return {'metrics': {'target': target, 'output': output.squeeze()}}

    def qual_step(self, batch, batch_idx):
        data, target = batch
        output = self.forward(data.clone())

        # We want to save the information here so that we can reconstruct the tile
        # see dataset/tools/callbacks/create_tile.py for more information
        
        xyzi = data[:, 0, :4]  # get the xyzi data from the corrupted gt-copy
        
        # append the corrupted version and the predicted intensity
        xyzi = torch.cat((xyzi, target.unsqueeze(1)), dim=1) 
        xyzi = torch.cat((xyzi, output.squeeze(-1)), dim=1)
        if self.xyzi is not None: 
            self.xyzi = torch.cat((self.xyzi, xyzi.cpu()))
        else:
            self.xyzi = xyzi.cpu()

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
        optimizer = Adam(self.net.parameters())
        scheduler = CyclicLR(
              optimizer,
              1e-8,
              1e-2,
              step_size_up=len(self.train_dataset)//self.batch_size//2,
              mode='triangular2',
              #scale_fn = lambda x: 1 / ((5/4.) ** (x-1)), # can't pickle this :\
              cycle_momentum=False)

        # scheduler = StepLR(optimizer, step_size=1)
        return [optimizer], [scheduler]


