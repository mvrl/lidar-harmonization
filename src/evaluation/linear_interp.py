from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import code
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.stats import gaussian_kde

from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CyclicLR
from src.dataset.tools.dataloaders import get_dataloader
from src.evaluation.tools.tools import HDataset

def interp2d(data, idx=1, method='linear', n_size=None):
    if n_size is None:
        n_size = len(data)
    return np.concatenate([
            griddata(
                n[idx:n_size+idx, :2],
                n[idx:n_size+idx, 3],
                n[0, :2], method=method
            ) for n in data])

def linear_interp(
	train_csv_path, 
	tile_csv_path,
	interpolation_method="linear", 
	harmonization_method="lstsq", 
        n_size=150,
	target_scan=1, 
	batch_size=50, 
	workers=8):
    
    train_csv_path = Path(train_csv_path)
    tile_csv_path = Path(tile_csv_path)
    train_dataloader = get_dataloader(train_csv_path, batch_size, workers, limit=100000)
    tile_dataloader = get_dataloader(tile_csv_path, batch_size, workers, drop_last=False)

    interp_func = interp2d
    
    running_loss = 0

    if Path(f"{n_size}_{interpolation_method}_interp.npy").exists():
        print("loaded data")
        dataset = np.load(f"{n_size}_{interpolation_method}_interp.npy")
    else:
        dataset = np.empty((0, 5))
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch_idx, batch in enumerate(pbar):
            data, h_target, i_target = batch
            try: 
                interpolation = interp_func(data, method=interpolation_method, n_size=n_size)
            except:
                print("skipped bad data")
                continue
                
            s_scan = data[:, 1, 8]
            t_scan = data[:, 0, 8]
	            
            new_batch = np.stack((
	        interpolation, 
	        i_target.numpy(), 
	        h_target.numpy(), 
	        s_scan.numpy(), 
	        t_scan.numpy())).T
	            
	    nans = np.where(np.isnan(interpolation))
	    new_batch = np.delete(new_batch, nans, axis=0)
	            
	    loss = np.mean(np.abs(new_batch[:, 0] - new_batch[:, 1]))
	            
	    running_loss += loss * batch_size
	    total_loss = running_loss / (((batch_idx+1)* batch_size))
	    pbar.set_postfix({
	        'icurr':f"{loss:.3f}", 
	        "itotl":f"{total_loss:.3f}"})
	            
	    dataset = np.concatenate((dataset, new_batch))
	    
	loss = np.mean(np.abs(dataset[:, 0] - dataset[:, 1]))
	print("Interpolation Loss: ", loss)
	np.save(f"{n_size}_{interpolation_method}_interp.npy", dataset)

    # Harmonization
    dataset_f = dataset[dataset[:, 4] == 1]  # filter out source-source scans
    sources = np.unique(dataset_f[:, 3])  # create list of source scans
    transforms = {}

    # plt.rcParams['figure.dpi'] = 150
    # fig, ax = plt.subplots(2, 4)

    for i, s in enumerate(sources):
        # this is horrible, why do this
        dataset_f_s = dataset_f[dataset_f[:, 3] == s]  # filter on source-target pair
        X = dataset_f_s[:, 0]  # pred interpolation for s
        #   dataset_f_s[:, 1]  # gt interpolation for s
        y = dataset_f_s[:, 2]  # gt harmonization for s
        
        if harmonization_method is "lstsq":

            A = np.vstack([X**3, X**2, X, np.ones(len(X))]).T
            v1, v2, v3, c = np.linalg.lstsq(A, y, rcond=None)[0]
            transforms[(int(s), 1)] = (v1, v2, v3, c)
            idx = X.argsort()
            X, y = X[idx], y[idx]

            
        elif harmonization_method is "MLP":

            # some options:
            epochs=2
            batch_size=50
            phases=['train']
            # phases=['train', 'val']

            if 'val' in phases:
                # create dataloader for training:
                split = len(dataset_f_s) - len(dataset_f_s)//8
                np.random.shuffle(dataset_f_s)
                train, val = dataset_f_s[:split], dataset_f_s[split:]
                train_dataset = HDataset(train)
                val_dataset = HDataset(val)
                dataloaders = {
                    'train': DataLoader(
                            train_dataset, 
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=8),

                    'val': DataLoader(
                            val_dataset,
                            batch_size=batch_size,
                            num_workers=8)
                }

            else:
                train_dataset = HDataset(dataset_f_s)

                dataloaders = {
                        'train':
                        DataLoader(
                            train_dataset,
                            batch_size = batch_size,
                            shuffle=True,
                            num_workers=8)}


            # network architecture
            net = nn.Sequential(
                    nn.Linear(1, 8),
                    nn.ReLU(),
                    # nn.Linear(16, 8),
                    # nn.Dro0pout(p=.5),
                    nn.Linear(8, 1))
            
            # intialize weights as identity to speed up convergence
            net[0].weight.data.copy_(torch.eye(8, 1))
            # net[1].weight.data.copy_(torch.eye(8, 16))
            net[2].weight.data.copy_(torch.eye(1, 8))
            
            gpu = torch.device('cuda:0')
            net = net.to(device=gpu).double()
            criterion = nn.SmoothL1Loss()
            optimizer=Adam(net.parameters())
            scheduler = CyclicLR(
                    optimizer, 
                    1e-6, 
                    1e-2, 
                    step_size_up=len(dataloaders['train'])//2,
                    mode='triangular2',
                    cycle_momentum=False)

            best_loss = 1000
            pbar1 = tqdm(range(epochs), total=epochs)
            for epoch in pbar1:
                for phase in phases:
                    if phase == 'train':
                        net.train()
                    else:
                        net.eval()

                    running_loss = 0.0
                    total = 0.0
                    # pbar2 = tqdm(
                    #         dataloaders[phase],
                     #        total=len(dataloaders[phase]),
                      #       leave=False,
                       #      desc=f"    {phase.capitalize()}: {epoch+1}/{epochs}")

                    for idx, batch in enumerate(dataloaders[phase]):
                        
                        optimizer.zero_grad()
                        with torch.set_grad_enabled(phase == 'train'):
                            inpt, target = batch
                            target = target.to(device=gpu)
                            # my_in = torch.stack((
                            #     inpt**3, 
                            #     inpt**2, 
                            #    inpt), dim=1)
                            inpt = inpt.to(device=gpu).reshape(-1, 1)
                            # code.interact(local=locals())
                            harmonization = net(inpt)

                            loss = criterion(harmonization.squeeze(), target)
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()
                                scheduler.step()


                        running_loss += loss.item() * batch_size
                        # pbar2.set_postfix({
                        #     "loss": f"{running_loss}/(idx+1):3f",
                        #    "lr": f"{optimizer.param_groups[0]['lr']:.2E}"})

                        total += batch_size
                    running_loss /= len(dataloaders[phase])
                    if phase == 'val':
                        if running_loss < best_loss:
                            best_loss = running_loss
                            # pbar1.set_description(f"Best Loss: {best_loss:.3f}")
                    pbar1.set_description(f"Loss: {running_loss:.3f}")
                            

            
            transforms[(int(s), 1)] = net
            # idx = X.argsort()
            # X, y = X[idx], y[idx]
            

            # These plots are used in the jupyter notebook
            #    see: src/dataset/notebooks/linear_interpolation.ipynb

            # xy = np.vstack([y, X])
            # z = gaussian_kde(xy)(xy)
            # idx = z.argsort()
            # X, y, z = X[idx], y[idx], z[idx]
            # ax.flat[i].scatter(X, y, c=z, s=6)
            # ax.flat[i].plot(X, transforms[(int(s), 1)].predict(X), 'r')
            # ax.flat[i].axis('off')
            # ax.flat[i].set_title(f"{int(s)}")
        else:
            exit(f"No method: {harmonization_method}")
            
    # plt.show()
    print("method:", harmonization_method)

    running_loss = 0
    with torch.no_grad():
        fixed_tile = np.empty((0, 11), dtype=np.float64)
        pbar = tqdm(tile_dataloader, total=len(tile_dataloader))
        for batch_idx, batch in enumerate(pbar):
            data, h_target, i_target = batch

            tile_data = data[:, 0, :].numpy()
            
            intensity = tile_data[:, 3]
            
            source_scan = int(data[0, 0, 8])
            t = transforms[(source_scan, target_scan)]
            
            if harmonization_method is "lstsq":
                fixed_intensity = (t[0]*(intensity**3)) + (t[1]*(intensity**2)) + (t[2]*intensity) + t[3]
            
            if harmonization_method is "MLP":
                t = t.eval()
                inpt = torch.tensor(intensity).reshape(-1,1).to(device=gpu)

                fixed_intensity = t(inpt).reshape(-1, 1).cpu().numpy()
                # code.interact(local=locals())
                
            tile_data = np.concatenate((
                tile_data[:, :3], # XYZ
                h_target.numpy().reshape(-1, 1), # h_target
                fixed_intensity.reshape(-1, 1),  # h_pred
                tile_data[:, 3].reshape(-1, 1),  # i_target
                tile_data[:, 4:]), axis=1)
            
            loss = np.mean(np.abs(tile_data[:, 4] - tile_data[:, 3]))
            
            running_loss += loss * batch_size
            total_loss = running_loss / (((batch_idx+1) * batch_size))
            pbar.set_postfix({
                "hcur": f"{float(loss):.3f}",
                "htot": f"{float(total_loss):.3f}"
            })
            
            fixed_tile = np.concatenate((fixed_tile, tile_data))

    print("Harmonization Loss: ", np.mean(np.abs(fixed_tile[:, 4] - fixed_tile[:, 3])))

    #np.savetxt(tile_csv_path.parents[0] / f"fixed_li_{n_size}_{interpolation_method}_{harmonization_method}.txt.gz", fixed_tile)

if __name__=="__main__":
    # This creates the main table
    """
    for n in [5, 20, 50, 100]:
        for h_method in ["MLP", "lstsq",]:
            for i_method in ["linear", "nearest", "cubic"]:
                print("running on n_size=%s, i_method=%s, h_method=%s" % (n, i_method, h_method))
                linear_interp(
                        "dataset/synth_crptn/150/train.csv",
                        "dataset/synth_crptn/big_tile_no_overlap/big_tile_dataset.csv",
                        interpolation_method=i_method,
                        harmonization_method=h_method,
                        n_size=n)
    """

    # shifted eval
    for n in [5, 20, 50, 100]:
        for h_method in ["MLP", "lstsq",]:
            for i_method in ["linear", "nearest", "cubic"]:
                print("running on n_size=%s, i_method=%s, h_method=%s" % (n, i_method, h_method))
                linear_interp(
                        "dataset/synth_crptn+shift/150/train.csv",
                        "dataset/synth_crptn+shift/big_tile_no_overlap/big_tile_dataset.csv",
                        interpolation_method=i_method,
                        harmonization_method=h_method,
                        n_size=n)

