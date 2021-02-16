import code
from src.training.train import train
from src.datasets.dublin.config import config as dublin_config
from src.training.config import config as train_config
from src.datasets.tools.dataloaders import get_dataloaders
# 
config = {
	'dataset': dublin_config,
	'train': train_config
}


# build dataloaders
dataloaders = get_dataloaders(config['dataset'], config['train'])

# build model
model = train(dataloaders, config['dataset'], config['train'])

# do something....