from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.stats import gaussian_kde
from sklearn.neural_network import MLPRegressor

from src.dataset.tools.dataloaders import get_dataloader


def interp2d(data, idx=1):
    return np.concatenate([
            griddata(
                n[idx:, :2],
                n[idx:, 3],
                n[0, :2]
            ) for n in data])

def interp3d(data, idx=1):
    return np.concatenate([
            LinearNDInterpolator(
                n[idx:, :3],
                n[idx:, 3]
           )(n[0, :3]) for n in data])


def linear_interp(
	train_csv_path, 
	tile_csv_path,
	harmonization_method="lstsq", 
	interpolation_dim=2, 
	target_scan=1, 
	batch_size=50, 
	workers=4):
    
    train_csv_path = Path(train_csv_path)
    tile_csv_path = Path(tile_csv_path)
    train_dataloader = get_dataloader(train_csv_path, batch_size, workers)
    tile_dataloader = get_dataloader(tile_csv_path, batch_size, workers, drop_last=False)

    if int(interpolation_dim) == 2: interp_func = interp2d
    if int(interpolation_dim) == 3: interp_func = interp3d
    if int(interpolation_dim) not in [2, 3]: exit(f"invalid interpolation dim: {interpolation_dim}")

    running_loss = 0
    with torch.no_grad():
        dataset = np.empty((0, 5))
        pbar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch_idx, batch in enumerate(pbar):
            data, h_target, i_target = batch
            
            interpolation = interp_func(data, idx=1)
            
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

    # Harmonization
    dataset_f = dataset[dataset[:, 4] == 1]  # filter out source-source scans
    sources = np.unique(dataset_f[:, 3])  # create list of source scans
    transforms = {}

    # plt.rcParams['figure.dpi'] = 150
    # fig, ax = plt.subplots(2, 4)

    for i, s in enumerate(sources):
        dataset_f_s = dataset_f[dataset_f[:, 3] == s]  # filter on source-target pair
        X = dataset_f_s[:, 0] # interpolated value at source for given s
        y = dataset_f_s[:, 2]                 # all gt harmonization values for S
        
        if harmonization_method is "lstsq":

            A = np.vstack([X**3, X**2, X, np.ones(len(X))]).T
            v1, v2, v3, c = np.linalg.lstsq(A, y, rcond=None)[0]
            transforms[(int(s), 1)] = (v1, v2, v3, c)
            idx = X.argsort()
            X, y = X[idx], y[idx]

            # These plots are used in the jupyter notebook
            #    see: src/dataset/notebooks/linear_interpolation.ipynb

            # xy = np.vstack([y, X])
            # z = gaussian_kde(xy)(xy)
            # idx = z.argsort()
            # X, y, z = X[idx], y[idx], z[idx]
            # ax.flat[i].scatter(X, y, c=z, s=6)
            # ax.flat[i].plot(X, v1*(X**3)+ v2*(X**2) + v3*X + c, 'r')
            # ax.flat[i].axis('off')
            # ax.flat[i].set_title(f"{int(s)}")
            
            
        elif harmonization_method is "MLP":
            transforms[(int(s), 1)] = MLPRegressor(
                (100, 100, 100), random_state=1, max_iter=300).fit(X, y)

            transforms[(int(s), 1)] = MLPRegressor((100, 100, 100)).fit(X, y)
            idx = X.argsort()
            X, y = X[idx], y[idx]
            

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
                new_intensity = t.predict(intensity).reshape(-1, 1)
                
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


    np.savetxt(tile_csv_path.parents[0] / f"fixed_li_{harmonization_method}.txt.gz", fixed_tile)

if __name__=="__main__":
    linear_interp(
            "dataset/150/train.csv",
            "dataset/big_tile_no_overlap/big_tile_dataset.csv")

