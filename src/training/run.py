import code
import pandas as pd
from pathlib import Path

from src.training.train import train
from src.datasets.dublin.config import config as dublin_config
from src.training.config import config as train_config
from src.datasets.dublin.tools import create_dublin_dataset
from src.datasets.tools.dataloaders import get_dataloaders
from src.evaluation.dl_interp import dl_interp_model

# not needed?
import torch
from src.harmonization.inet_pn1 import IntensityNet


config = {
	'dataset': dublin_config,
	'train': train_config
}

# strategy: 
# 1. collect all scans
scans = [f for f in Path(config['dataset']['scans_path']).glob("*.npy")]

# 2. select target scan(s)
target_scan = config['dataset']['target_scan']
learned_harmonizations = {}
for s in scans:
	if s.stem is target_scan:
		learned_harmonizations[s.stem] = "target"
	else:
		learned_harmonizations[s.stem] = None

# create_eval_tile(config['dataset'])

# while not done:
current_target = [k for (k,v) in learned_harmonizations.items() if v is not None]

# 3. build dataset of source scans overlapping target scan(s) - add curr index to master.csv?
# create_dataset(current_target, dublin_config)  # this takes a while due to the querying problem

# find the scans in master.csv that will be learned this iteration
df = pd.read_csv(Path(config['dataset']['save_path']) / "master.csv")
currently_learning = df.source_scan.unique().tolist()
print(f"Learning harmonizations for {currently_learning}")

# 4. train to harmonize initial dataset 
# build dataloaders
dataloaders = get_dataloaders(config)

training_dataloaders = {k:v for k,v in dataloaders.items() if k != "eval"}
# build model
# model = train(training_dataloaders, config['dataset'], config['train'])
gpu = torch.device('cuda:0')
n_size = config['train']['neighborhood_size']
model = IntensityNet(n_size, interpolation_method="pointnet").double().to(device=gpu)
model.load_state_dict(torch.load("results/5/5_epoch=0.pt"))

if 'eval' in dataloaders:
	# run on eval tile and report results
	print(f"Evaluating...")
	h_mae, i_mae = dl_interp_model(model, dataloaders['eval'], config)
	print(f"Harmonization MAE: {h_mae:.4f}")
	print(f"Interpolation MAE: {i_mae:.4f}")

code.interact(local=locals())

# harmonize the scans in currently_learning
harmonize(model, currently_learning, config) # TO DO

