import os
import time
import torch
import code
import numpy as np
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import gaussian_kde

from model import IntensityNet

def train(dataset, config, use_valid=True, baseline=False, use_tb=False):
    if baseline:
        print("making baseline!")
    start_time = time.time()
    sample_count = len(dataset)
    indices = list(range(sample_count))
    phases = ['train']

    if use_valid:
        phases.append('val')
        split = sample_count // 5
        valid_idx = np.random.choice(
            indices,
            size=split,
            replace=False)

        train_idx = list(set(indices) - set(valid_idx))
        dataset_sizes = {'train': len(train_idx), 'val': len(valid_idx)}
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        dataloaders = {
            loader: DataLoader(
                dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                sampler=sampler,
                drop_last=True
                )
            for (loader, sampler) in [('train', train_sampler), ('val', valid_sampler)]}

    else:
        train_idx = indices
        dataset_sizes = {'train': len(train_idx)}
        train_sampler = SubsetRandomSampler(train_idx)
        dataloaders = {
            loader: DataLoader(
                dataset,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                sampler=sampler,
                drop_last=True
                )
            for (loader, sampler) in [('train', train_sampler)]}

    print("Training samples: %s" % len(train_sampler))
    if use_valid:
        print("Validation samples: %s" % len(valid_sampler))
        print("Samples: %s, %s" % (sample_count, sample_count == len(train_sampler) + len(valid_sampler)))

    # Instantiate Model(s)
    model = IntensityNet().to(config.device).double()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())
    iterations = 0
    best_loss = 10e3
    for epoch in range(config.epochs):
        epoch_time = time.time()
        print('\nEpoch {}/{}'.format(epoch+1, config.epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            print(f"Starting {phase} phase")
            if phase == 'train':
                model.train()
            else:
                model.eval()
                predicted = []
                ground_truth = []

            running_loss = running_acc = total = 0.0

            for batch_idx, batch in enumerate(dataloaders[phase]):
                train_time = time.time()
                gt, alt, _, _ = batch
                # Get the intensity of the ground truth point
                # code.interact(local=locals())
                i_gt = alt[:,0,3].to(config.device)

                # get rid of the first point from the training sample
                alt = alt[:, 1:, :]
                
                if baseline:
                    # only use the nearest point
                    alt = alt[:, 0, :]
                    alt = alt.unsqueeze(2)

                alt = alt.transpose(1, 2).to(config.device)

                optimizer.zero_grad()

                iterations += 1

                # forward pass
                # track grad only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output, trans, trans_feat = model(alt)
                    loss = criterion(output, i_gt)
                
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    if phase == 'val':
                        predicted.append(output)
                        ground_truth.append(i_gt)

                running_loss += loss.item()*output.shape[0]
                # running_acc += (predicted == labels).sum().item()
                total += output.size(0)

                if batch_idx % 150 == 0 and phase == 'train':
                    print(f"[{batch_idx}/{len(dataloaders[phase])}] loss: {running_loss/total:.3f}")

            running_loss = running_loss/dataset_sizes[phase]
            # running_acc = running_acc/dataset_sizes[phase]

            print(f"{phase.capitalize()}: Loss: {running_loss:.3f}")

            if (phase == 'val'):
                if running_loss < best_loss:
                    print("New best loss! Saving model...")
                    torch.save(model.state_dict(), "intensity_dict")
                    best_loss = running_loss

                    # Create a KDE visualization of the trained network
                    
                    print("Generating KDE for visualization")
                    predicted = np.concatenate([t.cpu().detach().numpy() for t in predicted])
                    ground_truth = np.concatenate([t.cpu().detach().numpy() for t in ground_truth])
                        
                    # Calculate point density
                    xy = np.vstack([predicted, ground_truth])
                    z = gaussian_kde(xy)(xy)
                        
                    fig, ax = plt.subplots()
                    ax.scatter(predicted, ground_truth, c=z, s=100, edgecolor='')
                    plt.title("Predicted vs Actual")
                    plt.xlabel("predicted")
                    plt.ylabel("ground truth")
                    plt.savefig("evaluation/kde_validation.png")
                    
                    

    print(f"finished in {time.time() - start_time}")
    print("Training complete")        

    

