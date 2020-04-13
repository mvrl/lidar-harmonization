import time
import torch
import code
from pathlib import Path
import numpy as np
from dataset.lidar_dataset import LidarDataset
from torch.utils.data import SubsetRandomSampler, DataLoader
from torch.optim.lr_scheduler import CyclicLR
import torch.optim as optim
import torch.nn as nn

from model import IntensityNet
from util.metrics import Metrics, create_kde
from util.visdomviewer import VisdomLinePlotter

def train(config=None,
          dataset_csv=None,
          transforms=None,
          epochs=None,
          batch_size=None,
          use_valid=None,
          no_pass_center=None):

    start_time = time.time()
    dataset = LidarDataset(dataset_csv, transform=transforms) 
    sample_count = len(dataset)
    indices_list = list(range(sample_count))
    indices = {}



    suffix = dataset_csv.split("/")[1]
    if suffix.split("_")[0] == '0' and no_pass_center:
        exit("No points to train on! Use a bigger "
             "neighborhood or remove no_pass_center")

    if no_pass_center:
        save_suffix = "_nc"
    else:
        save_suffix = ""

    plotter = VisdomLinePlotter(f"{suffix}{save_suffix}")
        
    Path(f"results/{suffix}").mkdir(parents=True, exist_ok=True)
    phases = ['train']

    if use_valid:
        phases.append('val')
        split = sample_count // 5
        indices['val'] = np.random.choice(indices_list, size=split, replace=False)
        indices['train'] = list(set(indices_list) - set(indices['val']))

    else:
        indices['train'] = indices_list
    
    dataset_sizes = {phase : len(indices[phase]) for phase in phases}
    for phase in phases:
        print(f"{phase.capitalize()} samples: {dataset_sizes[phase]} => "
              f"{dataset_sizes[phase]//batch_size} iterations")
        
    samplers = {phase : SubsetRandomSampler(indices[phase]) for phase in phases}
    dataloaders = {
        phase : DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=config.num_workers,
            sampler=sampler,
            drop_last=True)
        for phase, sampler in samplers.items()}

    # Instantiate Model(s)
    model = IntensityNet(
        num_classes=config.num_classes
    ).to(config.device).double()
    
    optimizer = optim.Adam(model.parameters())
    iterations = 0; best_loss = 10e3
    lr_scheduler = CyclicLR(optimizer,
                            0.000001,
                            0.0001,
                            step_size_up=dataset_sizes['train']//batch_size//2,
                            mode='triangular',
                            cycle_momentum=False)

    metrics = Metrics(["mae"], None, batch_size)
    
    for epoch in range(epochs):
        epoch_time = time.time()
        print('\nEpoch {}/{}'.format(epoch+1, epochs)); print('-' * 10)

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

                if no_pass_center:
                    alt = alt[:, 1:, :]

                if len(alt.shape) == 2: 
                    alt = alt.unsqueeze(1)
                    
                alt = alt.transpose(1, 2).to(config.device)

                optimizer.zero_grad()
                iterations += 1

                with torch.set_grad_enabled(phase == 'train'):
                    x, _, _ = model(alt, fid)
                    loss = config.criterion(x.squeeze(), i_gt)
                    metrics.collect_metrics(x.detach().squeeze(), i_gt)
                
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                running_loss += loss.item()*x.shape[0]
                total += x.size(0)

                if batch_idx % ((len(samplers[phase])/batch_size)//10) == 0:
                    metrics.compute_metrics()
                    # code.interact(local=locals())
                    print(f"[{batch_idx}/{len(dataloaders[phase])}] "
                          f"loss: {running_loss/total:.3f} "
                          f"MAE: {metrics.metrics['mae']:.3f}")

                lr_scheduler.step()
                plotter.plot('lr', 'val', 'Learning Rate', iterations, lr_scheduler.get_lr()[0])            

            running_loss = running_loss/dataset_sizes[phase]

            print(f"{phase}: Loss {running_loss:.3f}")

            if (phase == 'val'):
                if running_loss < best_loss:
                    print("New best loss! Saving model...")

                    torch.save(model.state_dict(), f"results/{suffix}/model{save_suffix}.pt")
                    best_loss = running_loss

                    # Create a KDE plot for the network at this state
                    print("Generating KDE for visualization")

                    create_kde(metrics.targ_,
                               metrics.pred_,
                               "ground truths",
                               "predictions",
                               f"results/{suffix}/kde_validation{save_suffix}.png")

                    
            plotter.plot(f"{phase}_loss",'val',f"{phase.capitalize()} Loss", epoch, running_loss)


        print(f"Epoch finished in {(time.time() - epoch_time):.3f} seconds")
        metrics.clear_metrics()


            

    print(f"finished in {time.time() - start_time}")
    print("Training complete")        

