import code
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm, trange
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from src.training.forward_pass import forward_pass, get_metrics
from src.training.dataloaders import get_dataloader
from src.harmonization.model_npl import HarmonizationNet
from src.dataset.tools.metrics import create_interpolation_harmonization_plot as create_kde

import warnings
warnings.filterwarnings(action="ignore")
# High level training script

b_size=50
num_workers=6
n_size=5
epochs=50
gpu = torch.device('cuda:0')

print(f"Starting training with n_size {n_size}, b_size {b_size}")
results_path = Path(f"results/{n_size}")
results_path.mkdir(parents=True, exist_ok=True)
ckpt_path = None

csvs = {"train": "dataset/150/train.csv",
        "val": "dataset/150/val.csv"
        # "test": "dataset/150/test.csv"
        }


phases = [k for k in csvs]
dataloaders = {k: get_dataloader(v, b_size, num_workers) for k, v in csvs.items()}

model = HarmonizationNet(neighborhood_size=n_size).double() 
model.to(device=gpu)
criterions = {
        'harmonization': nn.SmoothL1Loss(), 
        'interpolation': nn.SmoothL1Loss()}

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
        
        data = []
        
        running_loss = 0.0
        total = 0.0
        pbar2 = tqdm(
                dataloaders[phase], 
                total=len(dataloaders[phase]), 
                leave=False,
                desc=f"   {phase.capitalize()}: {epoch+1}/{epochs}")

        for idx, batch in enumerate(pbar2):
            output = forward_pass(model, phase, batch, criterions, optimizer, scheduler, gpu)
            data.append(output)
            running_loss += output['loss'].item() * b_size
            pbar2.set_postfix({
                "loss" : f"{running_loss/(idx+1):.3f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2E}"})

            total += b_size
    
        running_loss /= len(dataloaders[phase])

        if phase == 'val':
            if running_loss < best_loss:
                new_ckpt_path = results_path / f"{n_size}_epoch={epoch}.pt"
                torch.save(model.state_dict(), new_ckpt_path)
                
                if ckpt_path:
                    ckpt_path.unlink() # delete previous checkpoint
                ckpt_path = new_ckpt_path

                best_loss = running_loss
                pbar1.set_description(f"Best Loss: {best_loss:.3f}")
                pbar1.set_postfix({"ckpt": f"{epoch}"})

                # create kde for val
                h_target, h_preds, h_mae, i_target, i_preds, i_mae = get_metrics(data)
                create_kde(
                    h_target, h_preds, h_mae, 
                    i_target, i_preds, i_mae, 
                    phase, results_path / f"val_kde_{n_size}.png")


