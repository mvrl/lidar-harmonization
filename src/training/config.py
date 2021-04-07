from collections import defaultdict
from src.config.project import Project
import torch
import os


p = Project()

def default_value():
    return ''

config = defaultdict(default_value)

config['name'] =  'dublin',

# Training settings
config['epochs'] = 15
config['batch_size'] = 50
config['neighborhood_size'] = 5
config['min_lr'] = 1e-6
config['max_lr'] = 1e-2
config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config['num_workers'] = int(os.getenv('SLURM_CPUS_PER_TASK', 8))

# GPU settings
config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config['num_gpus'] = int(torch.cuda.device_count())

# Output
config['results_path'] = str(p.root / f"results/{config['neighborhood_size']}")
config['model_save_path'] = str(p.root / f"models/{config['neighborhood_size']}")
config['tqdm'] = False  # True to disable progress bars
