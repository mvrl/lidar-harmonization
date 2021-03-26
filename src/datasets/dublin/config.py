from collections import defaultdict
from src.config.project import Project
from os import getenv
from pathlib import Path
import tempfile


p = Project()

def default_value():
    return ''

config = defaultdict(default_value)

config['name'] = 'dublin'

config['create_new'] = False

## directories
config['scans_path'] =  p.root / 'datasets/dublin/npy/'
#config['scans_path'] = str(p.root / 'datasets/dublin/test_npy/')

# Create a permanent dataset
config['save_path'] = p.root / 'datasets/dublin/150'
config['save_path'].mkdir(exist_ok=True, parents=True)

# Create the dataset temporarily (delete on process close)
# config['save_path_obj'] = tempfile.TemporaryDirectory(prefix="pipeline", dir="/tmp")
# config['save_path'] = Path(config['save_path_obj'].name)

config['hdf5_path'] = config['save_path'] / "dataset.h5"

config['plots_path'] = p.root/ 'datasets/dublin/150/plots'
config['plots_path'].mkdir(exist_ok=True, parents=True)

config['harmonized_path'] = p.root / 'datasets/dublin/harmonized'
config['harmonization_plots_path'] = p.root / 'datasets/dublin/harmonized/plots'
config['harmonization_plots_path'].mkdir(exist_ok=True, parents=True)

# Creation settings
config['target_scan'] =  '1'
config['igroup_size'] =  5
config['igroup_sample_size'] =  350  # needs to be higher for smaller collections
config['max_chunk_size'] =  500
config['max_n_size'] =  150
config['creation_log_conf'] = p.root / 'datasets/dublin/tools/logging.conf'
config['class_weights'] = p.root / 'datasets/dublin/150/class_weights.pt'
config['min_overlap_size'] = 200000
config['splits'] = {"train": .8, "test": .2}

config['workers'] = int(getenv('SLURM_CPUS_PER_TASK', 8))


# Eval tile 
config['eval_save_path'] =  p.root / 'datasets/dublin/150/eval_tile'
config['eval_tile_center'] = [316120.0, 234707.422, 1.749] # center of AOI
config['eval_source_scan'] = '39'
config['eval_tile_size'] = 1000000

# Corruption
config['dorf_path'] =  p.root / 'datasets/dublin/dorf.json'
config['mapping_path'] =  p.root / 'datasets/dublin/mapping.npy'
config['max_intensity'] =  512

# global shift settings:
config['bounds_path'] =  p.root / 'datasets/dublin/bounds.npy'
config['sig_floor'] =  .3
config['sig_center'] =  .5
config['sig_l'] =  100
config['sig_s'] =  .7

# training settings:
config['shift'] = False  # Apply global shift
config['use_ss'] = True  # use training examples from outside the overlap
config['phases'] = ['train', 'test']
config['dataloader_size'] = 100000

### do not modify
if config['shift']:
    config['shift_str'] = "_shift"

if not config['use_ss']:
    config['use_ss_str'] = "_ts"


config['tqdm'] = False  # True to disable tqdm bars
