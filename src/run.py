import code
import numpy as np
import pandas as pd
from pathlib import Path
import os
from src.training.train import train
from src.datasets.dublin.config import config as dublin_config
from src.training.config import config as train_config
from src.datasets.tools.create_dataset import create_dataset, create_eval_tile
from src.datasets.tools.dataloaders import get_dataloaders
from src.evaluation.dl_interp import dl_interp_model
from src.evaluation.harmonize import harmonize
from src.datasets.tools.harmonization_mapping import HarmonizationMapping


# TODO creating datasets for dublin and kylidar will be intrinsically 
#   different because the example creation will have gt for dublin but not
#   for kylidar. This might be a simple fix. 


config = {
    'dataset': dublin_config,
    'train': train_config
}

print("Starting up...")
print(f"PARTITION: {os.environ.get('SLURM_JOB_PARTITION')}")
print(f"Running with {config['dataset']['workers']} cores")
print(f"Found GPU {config['train']['device']}")

# build a mapping from source scan paths to target scan numbers.
hm = HarmonizationMapping(
    config['dataset']['scans_path'], 
    config['dataset']['target_scan'], 
    config['dataset']['harmonized_path'], 
    load_previous=False)

plots_path = Path(config['dataset']['harmonization_plots_path'])
plots_path.mkdir(exist_ok=True, parents=True)

# create evaluation tile for (optional) evaluation of models
if not (Path(config['dataset']['eval_save_path']) / 'eval_dataset.csv').exists():
    create_eval_tile(config['dataset'])

while True:
    # 3. build dataset of source scans overlapping target scan(s)
    create_dataset(hm, config['dataset'])

    if hm.done():
        break

    # 4. train to harmonize initial dataset
    dataloaders = get_dataloaders(config)
    training_dataloaders = {k:v for k,v in dataloaders.items() if k != "eval"}

    # build model
    print("building model")
    model, path = train(training_dataloaders, config)

    # harmonize the scans in harmonized_mapping if they aren't already
    # if 'eval' in dataloaders:
    #     # run on eval tile and report results
    #     print(f"Evaluating...")
    #     h_mae, i_mae = dl_interp_model(model, dataloaders['eval'], config)
    #     print(f"Harmonization MAE: {h_mae:.4f}")
    #     print(f"Interpolation MAE: {i_mae:.4f}")
    print("harmonizing")
    for source_scan_num in hm.get_stage(1):
        harmonized_scan = harmonize(model, 
                            hm[source_scan_num].source_scan_path.item(), 
                            hm[source_scan_num].harmonization_target.item(), 
                            config, sample_size=500000)

        np.save(str(hm.harmonization_path / (str(source_scan_num)+".npy")), harmonized_scan)
        hm.add_harmonized_scan_path(source_scan_num)
        hm.incr_stage(source_scan_num)
    if hm.done():
        break

print("finished")
hm.print_mapping()
