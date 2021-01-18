import code
from pathlib import Path
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from pptk import kdtree
from tqdm import tqdm

# from src.harmonization.model_npl import HarmonizationNet
from src.harmonization.inet_pn1 import IntensityNetPN1
from src.evaluation.histogram_matching import hist_match
from src.dataset.tools.apply_rf import ApplyResponseFunction
from src.dataset.tools.shift import get_physical_bounds, apply_shift_pc
from src.dataset.tools.dataloaders import transforms_no_load as transforms



class SimpleDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        if transforms is not None:
            return transforms(self.data[idx])
        else:
            return self.data[idx]

def fix_dublin_dl(dublin_path, dataset_csv, output_path=""):

    # prep output
    output_path = Path(output_path)
    if "shift" in dataset_csv:
        output_path = output_path / "fix_dublin_out" / "dl" / "shift"
    else:
        output_path = output_path / "fix_dublin_out" / "dl" / "default"

    output_path.mkdir(parents=True, exist_ok=True)
    
    # get scan paths
    dublin_path = Path(dublin_path)
    scans_paths = {f.stem : f.resolve() for f in dublin_path.glob("*.npy")}

    # Setup
    ARF = ApplyResponseFunction("dataset/dorf.json", "dataset/mapping.npy")
    bounds = get_physical_bounds(
        scans=dublin_path, 
        bounds_path="dataset/bounds.npy")
    sample_s = 100000
    batch_size = 50
    neighborhood_size=5

    # Load in dataset csv:
    df = pd.read_csv(dataset_csv)

    # grab all unique sources used for training:
    sources = df.source_scan.unique()
    print("Sources: ", sources)

    targets = df.target_scan.unique()
    target_scan_num = list(set(targets) - set(sources))[0]
    target_scan = scans_paths[str(target_scan_num)]
    print("Target: ", target_scan)
    target_camera = int(target_scan.stem)
    source_scans = [ scans_paths[str(source)] for source in sources ]
    
    # # load target as reference
    # ref = np.load(target_scan)

    # if "shift" in dataset_csv:
    #     print("applying shift")
    #     ref = apply_shift_pc(ref,
    #                          bounds[0][0],
    #                          bounds[0][1])
        
    # # save target to view in fix later
    # sample = np.random.choice(len(ref), sample_s)
    # ref_v = ref[sample].copy()
    # ref_v = np.concatenate((
    #         ref_v[:, :3],
    #         ref_v[:, 3].reshape(-1, 1),
    #         ref_v[:, 3].reshape(-1, 1),
    #         ref_v[:, 3].reshape(-1, 1),
    #         np.ones(len(ref_v)).reshape(-1, 1)
    #     ), axis=1)

    print("Saving reference: ", output_path / "ref.txt.gz")
    np.savetxt(str(output_path / "ref.txt.gz"), ref_v)
    
    # Load the model and place in evaluation mode
    device = torch.device("cuda:0")
    model = HarmonizationNet(neighborhood_size=neighborhood_size).to(device=device, dtype=torch.float64)
    model.load_state_dict(torch.load("results/0/0_epoch=23.pt"))
    model.eval()
    print("Model Loaded")

    for s in source_scans:
        path = Path(s)
        source = np.load(s)
        
        if "shift" in dataset_csv:
            print("applying shift")
            source = apply_shift_pc(source, 
                                    bounds[0][0], 
                                    bounds[0][1])
    
        # build a kd tree over the source.
        kd = kdtree._build(source[:, :3])
        query = kdtree._query(
            kd, 
            source[:, :3], 
            k=50, 
            dmax=1)
        
        # Confirm that a large majority of examples are reasonable. Use the good
        #   examples instead of the bad ones. 
        my_query = []
        for q in query:
            if len(q) == neighborhood_size:
                my_query.append(q)

        good_sample_ratio = ((len(query) - len(my_query))/len(query)) * 100
        print(f"Found {good_sample_ratio:.3f}% of points with not enough close neighbors")   

        # sample from good examples
        sample = np.random.choice(len(my_query), sample_s)

        print("building dataset from queried points")
        dataset = []
        for i in tqdm(sample):

            gt_neighborhood = source[my_query[i]]
            alt_neighborhood = ARF(gt_neighborhood, int(path.stem), 512)
  
            example = np.concatenate((
                np.expand_dims(gt_neighborhood[0, :], 0),
                np.expand_dims(alt_neighborhood[0, :], 0),
                alt_neighborhood))
            
            dataset.append(example)

        dataset = np.stack(dataset)
        dataset = dataset[sample]
        dataset = SimpleDataset(dataset, 
                                transforms=transforms)

        dataloader = DataLoader(dataset, 
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=8,
                                drop_last=False)
        
        # iterate over batches
        fixed_source = np.empty((0, 12), dtype=np.float64)
        for batch in dataloader:
            with torch.no_grad():
                # format example
                data, h_target, i_target = batch

                scan_data = data[:, 0, :].numpy()
                data = data.to(device=device)                
                
                # specify that we want to harmonize to `target_camera`
                data[:, 0, -1] = target_camera

                harmonization, interpolation, _ = model(data)

                scan_data = np.concatenate((
                    scan_data[:, :3],
                    h_target.numpy().reshape(-1, 1),
                    i_target.numpy().reshape(-1, 1),
                    harmonization.cpu().numpy(),
                    interpolation.cpu().numpy(),
                    scan_data[:, 5:],
                    np.zeros(len(h_target)).reshape(-1, 1)
                    ), axis=1)

                fixed_source = np.concatenate((fixed_source, scan_data))

        mae = np.mean(np.abs(np.clip(scan_data[:, 5], 0, 1), scan_data[:, 3]/512))
        print(path.stem, ": MAE", mae)
        print("Saving: ", output_path / (str(path.stem)+".txt.gz"))
        np.savetxt(output_path / (str(path.stem)+".txt.gz"), fixed_source)

if __name__ == "__main__":
    fix_dublin_dl("dataset/dublin/npy", "dataset/synth_crptn/150/train.csv")