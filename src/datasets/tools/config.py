from collections import defaultdict
from src.config.project import Project
from os import getenv
from pathlib import Path
import tempfile

# this it the base config! 

p = Project()

def default_value():
    return ''

config = defaultdict(default_value)

config['name'] = 'my_lidar_collection'

config['create_new'] = True
config['data_path'] = p.root / f"datasets/{config['name']}/data"

## directories
config['scans_path'] =  config['data_path'] / 'npy'
# config['scans_path'] = config['data_path'] / 'test_npy'

# Create a permanent dataset
config['save_path'] = config['data_path'] / '150'
config['save_path'].mkdir(exist_ok=True, parents=True)

# Create the dataset temporarily (delete on process close)
# config['save_path_obj'] = tempfile.TemporaryDirectory(prefix="pipeline", dir="/tmp")
# config['save_path'] = Path(config['save_path_obj'].name)

config['dataset_path'] = config['save_path'] / "dataset.h5"

config['plots_path'] = config['save_path'] / 'plots'
config['plots_path'].mkdir(exist_ok=True, parents=True)

config['harmonized_path'] = config['data_path'] / 'harmonized'
config['harmonization_plots_path'] = config['harmonized_path'] / 'plots'
config['harmonization_plots_path'].mkdir(exist_ok=True, parents=True)

# Creation settings
config['target_scan'] =  '1'
config['igroup_size'] =  0.01
config['igroup_sample_size'] =  500
config['max_chunk_size'] =  500
config['max_n_size'] =  150
config['creation_log_conf'] = p.root / 'datasets/dublin/tools/logging.conf'
config['class_weights'] = config['save_path'] / 'class_weights.pt'
config['min_overlap_size'] = 200000
config['splits'] = {"train": .8, "test": .2}
config['workers'] = int(getenv('SLURM_CPUS_PER_TASK', 8))

# training settings:
config['use_ss'] = True  # use training examples from outside the overlap
config['phases'] = ['train', 'test']
config['shift'] = True  # Apply global shift
config['dataloader_size'] = 500000

# this sets the limit for RAM usage during the actual harmonization process. 
#   Using a higher value consumes more RAM. Recommend 100-200k for 32GB of RAM.
config['dataloader_size'] = 10000

config['tqdm'] = False  # True to disable tqdm bars
