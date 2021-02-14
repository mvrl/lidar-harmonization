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
from src.datasets.tools.dataloaders import get_dataloader
from src.evaluation.tools import HDataset, mlp_train, mlp_inference, lstsq_method

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
    workers=8,
    interp_data_path="temp_default"
):

    gpu = torch.device('cuda:0')
    
    train_csv_path = Path(train_csv_path)
    tile_csv_path = Path(tile_csv_path)
    train_dataloader = get_dataloader(train_csv_path, batch_size, workers, limit=1000000)
    tile_dataloader = get_dataloader(tile_csv_path, batch_size, workers, drop_last=False)

    interp_func = interp2d
    
    running_loss = 0
    interp_data_path = Path(interp_data_path)
    interp_data_path.mkdir(parents=True, exist_ok=True)

    if (interp_data_path / f"{n_size}_{interpolation_method}_interp.npy").exists():
        # print("loaded data")
        dataset = np.load(str(interp_data_path / f"{n_size}_{interpolation_method}_interp.npy"))
    else:
        dataset = np.empty((0, 5))
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch_idx, batch in enumerate(pbar):
            data, h_target, i_target = batch
            try: 
                interpolation = interp_func(data, method=interpolation_method, n_size=n_size)
            except:
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
        print("Interpolation Error: ", loss)
        np.save(interp_data_path / f"{n_size}_{interpolation_method}_interp.npy", dataset)

    # Harmonization        
    if harmonization_method is "lstsq":
        model = lstsq_method(dataset, target_scan=target_scan)

    elif harmonization_method is "MLP":
        model = mlp_train(dataset, 30, batch_size, gpu)
        model.eval()
        
    else:
        exit(f"No method: {harmonization_method}")
            
    # Test method on the evaluation tile
    # print("method:", harmonization_method)

    running_loss = 0
    with torch.no_grad():
        fixed_tile = np.empty((0, 11), dtype=np.float64)
        pbar = tqdm(tile_dataloader, total=len(tile_dataloader))
        for batch_idx, batch in enumerate(pbar):
            data, h_target, i_target = batch
            # data is [batch, neighborhood, channels]
            
            
            if harmonization_method is "lstsq":
                source_intensity = data[:, 0, 3]
                source_scan = int(data[0, 0, 8])

                t = model[(source_scan, target_scan)]
                fixed_intensity = (t[0]*(source_intensity**3)) + (t[1]*(source_intensity**2)) + (t[2]*source_intensity) + t[3]
            
            if harmonization_method is "MLP":
                with torch.set_grad_enabled(False):   

                    i_center = data[:, 1, 3]
                    h_target_t = h_target.clone()

                    source = data[:, 0, 8]
                    target = torch.tensor(target_scan).repeat(len(source)).double()

                    new_batch = torch.stack((i_center, 
                                             i_target, 
                                             h_target_t,
                                             source,
                                             target)).T

                    new_batch = new_batch.to(device=gpu)

                    fixed_intensity, h_target_t = model(new_batch)
                    fixed_intensity = fixed_intensity.cpu().numpy().squeeze()
                                                              
            tile_data = np.concatenate((
                data[:, 0, :3], # XYZ
                h_target.numpy().reshape(-1, 1), # h_target
                fixed_intensity.reshape(-1, 1),  # h_pred
                data[:, 0, 3].reshape(-1, 1),    # i_target
                data[:, 0, 4:]), axis=1)
            
            loss = np.mean(np.abs(tile_data[:, 4] - tile_data[:, 3]))
            
            running_loss += loss * batch_size
            total_loss = running_loss / (((batch_idx+1) * batch_size))
            pbar.set_postfix({
                "hcur": f"{float(loss):.3f}",
                "htot": f"{float(total_loss):.3f}"
            })
            
            fixed_tile = np.concatenate((fixed_tile, tile_data))
    mae = np.mean(np.abs(fixed_tile[:, 4] - fixed_tile[:, 3]))
    print("Harmonization Error: ", mae)
    print("-"*80)

    np.savetxt(tile_csv_path.parents[0] / f"fixed_li_{n_size}_{interpolation_method}_{harmonization_method}_{mae:.3f}.txt.gz", fixed_tile)

if __name__=="__main__":
    # This creates the main table
    
    for i_method in ["linear", "cubic"]:
        continue
        for h_method in ["lstsq", "MLP"]:
            for n in [5, 20, 50, 100]:
                print(f"Running: {i_method} {h_method} {n} (no global shift)")

                linear_interp(
                    "dataset/synth_crptn/150/train.csv",
                    "dataset/synth_crptn/big_tile_no_overlap/big_tile_dataset.csv",
                    interpolation_method=i_method,
                    harmonization_method=h_method,
                    n_size=n)
                print("*"*80)

    for i_method in ["linear", "cubic"]:
        continue
        for h_method in ["lstsq", "MLP"]:
            for n in [5, 20, 50, 100]:
                print(f"Running: {i_method} {h_method} {n} (with global shift)")
                
                linear_interp(
                    "dataset/synth_crptn+shift/150/train.csv",
                    "dataset/synth_crptn+shift/big_tile_no_overlap/big_tile_dataset.csv",
                    interpolation_method=i_method,
                    harmonization_method=h_method,
                    n_size=n,
                    interp_data_path="temp_shift")
                print("*"*80)

    print("Running Nearest-interpolation")  # neighborhood size is irrelevant 
    for h_method in ["lstsq", "MLP"]:
        continue
        print(f"Running: nearest {h_method} (with global shift)")
        linear_interp(
            "dataset/synth_crptn/150/train.csv",
            "dataset/synth_crptn/big_tile_no_overlap/big_tile_dataset.csv",
            interpolation_method="nearest",
            harmonization_method=h_method,
            n_size=3)

    for h_method in ["lstsq", "MLP"]:
        print(f"Running: nearest {h_method} (with global shift)")
        linear_interp(
            "dataset/synth_crptn+shift/150/train.csv",
            "dataset/synth_crptn+shift/big_tile_no_overlap/big_tile_dataset.csv",
            interpolation_method="nearest",
            harmonization_method=h_method,
            n_size=3,
            interp_data_path="temp_shift"
        )

