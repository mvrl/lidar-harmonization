import code
import pandas as pd
from pathlib import Path
from shutil import copyfile

from src.training.train import train
from src.datasets.dublin.config import config as dublin_config
from src.training.config import config as train_config
from src.datasets.dublin.tools import create_dublin_dataset
from src.datasets.tools.create_dataset import create_dataset
from src.datasets.tools.dataloaders import get_dataloaders
from src.evaluation.dl_interp import dl_interp_model
from src.evaluation.harmonize import harmonize

# not needed?
import torch
from src.harmonization.inet_pn1 import IntensityNet
from pprint import pprint


config = {
	'dataset': dublin_config,
	'train': train_config
}

# strategy: 
# 1. collect all scans
scans = [f for f in Path(config['dataset']['scans_path']).glob("*.npy")]

# 2. select target scan(s)
target_scan = config['dataset']['target_scan']
target_scan_path = Path(config['dataset']['scans_path']) / (target_scan+".npy")
# copy this to the harmonized directory. All "target scans" will be loaded
#   from this directory. 

copyfile(str(target_scan_path), str(Path(config['dataset']['harmonized_path']) / (target_scan+".npy")))

harmonization_mapping = {}

# get all the scans in place
for s in scans:
	if s.stem is target_scan:
		harmonization_mapping[s] = int(target_scan)
	else:
		harmonization_mapping[s] = None

while None in harmonization_mapping.values():

	# CRAP. creating datasets for dublin and kylidar will be
	#   intrinsically different because you're including the gt 
	#   for dublin. This might be a simple fix. 

	# 3. build dataset of source scans overlapping target scan(s) - add curr index to master.csv?
	create_dataset(
		harmonization_mapping, 
		config['dataset'])  # this takes a while due to the querying problem
	
	# create_eval_tile(config['dataset'])

	# find the scans in master.csv that will be learned this iteration
	#   specifically, mapping needs to be created that defines which sources
	#   source scans are harmonized to which target scan(s)

	df = pd.read_csv(Path(config['dataset']['save_path']) / "master.csv")
	unique = df.groupby(['source_scan', 'target_scan']).size().reset_index().rename(columns={0:'count'})
	unique_no_ss = unique.loc[unique.source_scan != unique.target_scan]

	# if harmonization_mappping[s].stem exists in unique_no_ss sources, update with target info
	for source_path, target in harmonization_mapping.items():
		if target is "target":
			continue
		if int(source_path.stem) in unique_no_ss.source_scan.values:
			harmonization_mapping[source_path] = unique_no_ss.loc[
				unique_no_ss.source_scan == int(source_path.stem)].target_scan.item()


	print(f"Harmonization Mapping:")
	pprint({int(k.stem):v for k,v in harmonization_mapping.items()})
	code.interact(local=locals())

	# 4. train to harmonize initial dataset 
	# build dataloaders
	dataloaders = get_dataloaders(config)
	training_dataloaders = {k:v for k,v in dataloaders.items() if k != "eval"}

	# build model
	# model, path = train(training_dataloaders, config['dataset'], config['train'])
	gpu = torch.device('cuda:0')
	n_size = config['train']['neighborhood_size']
	model = IntensityNet(n_size, interpolation_method="pointnet").double().to(device=gpu)
	model.load_state_dict(torch.load("results/5/5_epoch=4.pt"))

	# remove False in future
	if False and 'eval' in dataloaders:
		# run on eval tile and report results
		print(f"Evaluating...")
		h_mae, i_mae = dl_interp_model(model, dataloaders['eval'], config)
		print(f"Harmonization MAE: {h_mae:.4f}")
		print(f"Interpolation MAE: {i_mae:.4f}")


	# harmonize the scans in currently_learning
	harmonized_path = Path(config['dataset']['harmonized_path'])
	harmonized_path.mkdir(exist_ok=True, parents=True)
	plots_path = Path(config['dataset']['harmonization_plots_path'])
	plots_path.mkdir(exist_ok=True, parents=True)

	# need a mapping of scan_path to target scan number
	for source_scan_path, target_scan_num in harmonization_mapping.items():
		harmonize(model, source_scan_path, target_scan_num, config)

