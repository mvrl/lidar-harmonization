import os
import time
import torch
import code
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch.optim as optim
import torch.nn as nn

from model import IntensityNet
from util.metrics import Metrics, create_kde

def train(dataset=None, config=None, use_valid=True, use_tb=False):

    start_time = time.time()
    sample_count = len(dataset)
    indices_list = list(range(sample_count))
    indices = {}

    phases = ['train']

    if use_valid:
        phases.append('val')
        split = sample_count // 5
        indices['val'] = np.random.choice(indices_list, size=split, replace=False)
        indices['train'] = list(set(indices_list) - set(indices['val']))

    else:
        indices['train'] = indices_list
    
    dataset_sizes = {phase : len(indices[phase]) for phase in phases}
    samplers = {phase : SubsetRandomSampler(indices[phase]) for phase in phases}

    dataloaders = {
        phase : DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            sampler=sampler,
            drop_last=True)
        for phase, sampler in samplers.items()}

    # Instantiate Model(s)
    model = IntensityNet().to(config.device).double()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters())
    iterations = 0; best_loss = 10e3

    metrics = Metrics(["mae"], None)
    
    for epoch in range(config.epochs):
        epoch_time = time.time()
        print('\nEpoch {}/{}'.format(epoch+1, config.epochs)); print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            print(f"Starting {phase} phase")
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = total = 0.0

            for batch_idx, batch in enumerate(dataloaders[phase]):
                train_time = time.time()
                gt, alt, fid, _ = batch

                # Get the intensity of the ground truth point
                i_gt = gt[:,0,3].to(config.device)
                fid = fid.to(config.device)

                # get rid of the first point from the training sample
                alt = alt[:, 1:, :]
                alt = alt.transpose(1, 2).to(config.device)

                optimizer.zero_grad()
                iterations += 1

                with torch.set_grad_enabled(phase == 'train'):
                    output, _, _ = model(alt, fid)
                    metrics.collect(output, i_gt)
                    loss = criterion(output, i_gt)
                
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                running_loss += loss.item()*output.shape[0]
                total += output.size(0)

                if batch_idx % ((len(samplers[phase])/config.batch_size)//10) == 0:
                    metrics.compute()
                    print(f"[{batch_idx}/{len(dataloaders[phase])}] "
                          f"loss: {running_loss/total:.3f} "
                          f"MAE: {metrics.metrics['mae']:.3f} ")

            running_loss = running_loss/dataset_sizes[phase]

            print(f"{phase.capitalize()}: Loss {running_loss:.3f}")

            if (phase == 'val'):
                if running_loss < best_loss:
                    print("New best loss! Saving model...")
                    torch.save(model.state_dict(), "intensity_dict")
                    best_loss = running_loss

                    # Create a KDE plot for the network at this state
                    print("Generating KDE for visualization")
                    create_kde(metrics.pred_, metrics.targ_, "evaluation/kde_validation.png")

            metrics.clear_metrics()
        print(f"Epoch finished in {(time.time() - epoch_time):.3f} seconds")
            

    print(f"finished in {time.time() - start_time}")
    print("Training complete")        

