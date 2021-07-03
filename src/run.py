import code
import numpy as np
import pandas as pd
from pathlib import Path
import os

from src.training.train import train
# from src.datasets.<my_dataset>.config import config as dataset_config
from src.datasets.dublin.config import config as dataset_config
from src.training.config import config as train_config
from src.datasets.tools.create_dataset import create_dataset, create_eval_tile
from src.datasets.tools.dataloaders import get_dataloaders
from src.evaluation.dl_interp import dl_interp_model
from src.evaluation.harmonize import harmonize
from src.datasets.tools.harmonization_mapping import HarmonizationMapping

config = {
    'dataset': dataset_config,
    'train': train_config
}

# build a mapping from source scan paths to target scan numbers.
hm = HarmonizationMapping(config)

# create evaluation tile for (optional) evaluation of models
if not (Path(config['dataset']['eval_dataset']).exists()):
    create_eval_tile(config['dataset'])
else:
    print("Found eval tile")

print("Starting up...")
partition = os.getenv('SLURM_JOB_PARTITION', None)
print(f"Partition: {os.environ.get('SLURM_JOB_PARTITION')}")
print(f"Running with {config['dataset']['workers']} cores")
print(f"Found GPU {config['train']['device']}")
print(f"Using {config['train']['num_gpus']} GPUs")


while True:
    # 3. build dataset of source scans overlapping target scan(s)
    print("building dataset")
    create_dataset(hm, config['dataset'])

    if hm.done():
        break

    dataloaders = get_dataloaders(config)
    training_dataloaders = {k:v for k,v in dataloaders.items() if k != "eval"}

    print("building model")
    model, path = train(training_dataloaders, config)

    # harmonize eval tile
    # harmonize_eval_tile(model, config)

    # perhaps it makes sense to add `model_path` to the mapping csv so we can load it back in later
    for source_scan_num in hm.get_stage(1):
        print(f"Harmonizing scan {source_scan_num}")
        
        harmonized_scan = harmonize(model,
                            hm[source_scan_num].source_scan_path.item(),
                            hm[source_scan_num].harmonization_target.item(),
                            config)

        np.save(str(hm.harmonization_path / (str(source_scan_num)+".npy")), harmonized_scan)
        hm.add_harmonized_scan_path(source_scan_num)
        hm.incr_stage(source_scan_num)

    if hm.done():
        break

# if hm.done():
#   repackage the harmonized scans as laz

print("finished")
hm.print_mapping()
