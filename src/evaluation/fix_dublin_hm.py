# Compare methods on dublin on a large scale. This script generates fixed 
#   versions of the source scans for each source in the training set. These 
#   can then be analyzed later. 

import code
from pathlib import Path
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from pptk import kdtree
from tqdm import tqdm

from src.harmonization.model_npl import HarmonizationNet
from src.evaluation.histogram_matching import hist_match
from src.dataset.tools.apply_rf import ApplyResponseFunction
from src.dataset.tools.shift import get_physical_bounds, apply_shift_pc
from src.dataset.tools.dataloaders import transforms_no_load as transforms



def fix_dublin_hm(dublin_path, dataset_csv, output_path=""):

    # prep output
    output_path = Path(output_path)
    if "shift" in dataset_csv:
        output_path = output_path / "fix_dublin_out" / method / "shift"
    else:
        output_path = output_path / "fix_dublin_out" / method / "default"

    output_path.mkdir(parents=True, exist_ok=True)
    
    # get scan paths
    dublin_path = Path(dublin_path)
    scans_paths = {f.stem : f.resolve() for f in dublin_path.glob("*.npy")}

    # Setup
    ARF = ApplyResponseFunction("dataset/dorf.json", "dataset/mapping.npy")
    bounds = get_physical_bounds(
        scans=dublin_path, 
        bounds_path="dataset/bounds.npy")

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
    
    # load target as reference
    ref = np.load(target_scan)

    if "shift" in dataset_csv:
        ref = apply_shift_pc(ref,
                             bounds[0][0],
                             bounds[0][1])
        
    # save target to view in fix later
    sample = np.random.choice(len(ref), sample_s)
    ref_v = ref[sample].copy()
    ref_v = np.concatenate((
            ref_v[:, :3],
            ref_v[:, 3].reshape(-1, 1),
            ref_v[:, 3].reshape(-1, 1),
            ref_v[:, 3].reshape(-1, 1),
            np.ones(len(ref_v)).reshape(-1, 1)
        ), axis=1)

    print("Saving reference: ", output_path / "ref.txt.gz")
    np.savetxt(str(output_path / "ref.txt.gz"), ref_v)

    for s in source_scans:
        path = Path(s)
        source = np.load(s)

        sample = np.random.choice(len(source), sample_s)
        
        if "shift" in dataset_csv:
            print("applying shift")
            source = apply_shift_pc(source, 
                                    bounds[0][0], 
                                    bounds[0][1])
        
        source = source[sample]
        alt_source = ARF(source, int(path.stem), 512)

        gt_intensity = source[:, 3].copy()
        alt_intensity = alt_source[:, 3].copy()
        fix_intensity = hist_match(alt_intensity, ref[:, 3])

        fix_source = source.copy()
        fix_source = np.concatenate((
            fix_source[:, :3],
            gt_intensity.reshape(-1, 1),
            alt_intensity.reshape(-1, 1),
            fix_intensity.reshape(-1, 1),
            np.zeros(len(gt_intensity)).reshape(-1, 1) # denote source scans
        ), axis=1)
        
        mae = np.mean(np.abs(np.clip(fix_intensity, 0, 1), gt_intensity/512))
        print(path.stem, ": MAE", mae)
        print("Saving: ", output_path / (str(path.stem)+".txt.gz"))
        np.savetxt(output_path / (str(path.stem)+".txt.gz"), fix_source)