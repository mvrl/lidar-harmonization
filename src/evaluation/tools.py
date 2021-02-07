import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
import code
import numpy as np
from tqdm import tqdm

class HDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class SimpleIntensityMLP(nn.Module):
    def __init__(self, embed_dim=3, num_classes=1, hidden_size=100):
        # this is just the harmonization section of the DL Interp method
        super(SimpleIntensityMLP, self).__init__()

        self.camera_embed = nn.Embedding(45, embed_dim)
        self.harmonization = nn.Sequential(
                nn.Linear(num_classes+embed_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes))

        self.harmonization[0].weight.data.copy_(torch.eye(hidden_size, 1+embed_dim))
        self.harmonization[2].weight.data.copy_(torch.eye(1, hidden_size))

    def forward(self, x):
        # [I_p, I_t, H_t, S, T] -> [H_p]
        
        # Camera embedding (no neighborhoods in this case)
        source_camera = x[:, 3].long()
        target_camera = x[:, 4].long()
   
        source_camera_embed = self.camera_embed(source_camera)
        target_camera_embed = self.camera_embed(target_camera)
        camera_info = source_camera_embed - target_camera_embed

        train_data = x[:, 0]  # interpolated data
        gt = x[:, 2]
        

        # create feature vector
        feat_vector = torch.cat((train_data.unsqueeze(1), camera_info.double()), dim=1)

        # return predicted harmonization value, gt harmonized value

        h = self.harmonization(feat_vector), gt

        return h

def mlp_train(dataset, epochs, batch_size, device=torch.device("cpu")):


    phases = ['train']

    
    # these just repackage data from the usual dataloaders, so all 
    #     transformations should remain intact. 

    train_dataset = HDataset(dataset)
    dataloaders = {
            'train':
            DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=8)}

    model = SimpleIntensityMLP().double()
    model = model.to(device=device)

    criterion = nn.SmoothL1Loss()
    optimizer = Adam(model.parameters())

    scheduler = CyclicLR(
        optimizer, 
        1e-6, 
        1e-2, 
        step_size_up=len(dataloaders['train'])//2,
        # mode='triangular2',
        scale_fn = lambda x: 1 / ((5/4.) ** (x-1)),
        cycle_momentum=False)

    best_loss = 1000
    pbar1 = tqdm(range(epochs), total=epochs, desc=f"Best Loss: {best_loss}")
    for epoch in pbar1:
        for phase in phases:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            total = 0.0

            for idx, batch in enumerate(dataloaders[phase]):
                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()
                    data = batch.to(device=device)
                    harmonization, target = model(data)

                    loss = criterion(harmonization.squeeze(), target.squeeze())
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        scheduler.step()  # step per batch w/ CyclicLR

                running_loss += loss.item() * batch_size

                total += batch_size
            running_loss /= len(dataloaders[phase])
            pbar1.set_description(f"Loss: {running_loss:.3f}")
            
    return model

def mlp_inference(model, data):
    model.eval()

    with torch.set_grad_enabled(False):
        return model(data)
        

def lstsq_method(dataset, target_scan=1):
    dataset_f = dataset[dataset[:, 4] == target_scan]  # filter out source-source scans
    sources = np.unique(dataset_f[:, 3])     # create list of source scans
    transforms = {}

    # Approximate AX + b = Y, where X is the source intensity and Y is the target
    #    Do this for each source. 
    for i, s in enumerate(sources):
        dataset_f_s = dataset_f[dataset_f[:, 3] == s]  # filter on source-target pair
        X = dataset_f_s[:, 0]  # pred interpolation
        y = dataset_f_s[:, 2]  # harmonization target

        # Naive basis expansion
        A = np.vstack([X**3, X**2, X, np.ones(len(X))]).T
        v1, v2, v3, c = np.linalg.lstsq(A, y, rcond=None)[0]
        transforms[(int(s), target_scan)] = (v1, v2, v3, c)

    return transforms

