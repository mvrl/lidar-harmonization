from collections import defaultdict
from src.config.project import Project

p = Project()

def default_value():
    return ''

config = defaultdict(default_value)


config['name'] =  'dublin'

# Creation settings
config['target_scan'] =  '1'
config['igroup_size'] =  5
config['igroup_sample_size'] =  500
config['max_chunk_size'] =  int(4e6)
config['max_n_size'] =  150
config['scans_path'] =  str(p.root / 'datasets/dublin/npy/')
config['save_path'] =  str(p.root / 'datasets/dublin/150')
config['class_weights'] = str(p.root / 'datasets/dublin/150/weights.npy')
config['min_overlap_size'] =  200000

# Corruption
config['dorf_path'] =  str(p.root / 'datasets/dublin/dorf.json')
config['mapping_path'] =  str(p.root / 'datasets/dublin/mapping.npy')
config['max_intensity'] =  512

# global shift settings:
config['bounds_path'] =  str(p.root / 'datasets/dublin/bounds.npy')
config['sig_floor'] =  .3
config['sig_center'] =  .5
config['sig_l'] =  100
config['sig_s'] =  .7

# training settings:
config['shift'] = False  # Apply global shift
config['use_ss'] = True  # use training examples from outside the overlap
config['phases'] = ['train', 'val']

### do not modify
if config['shift']:
    config['shift_str'] = "_shift"

if not config['use_ss']:
    config['use_ss_str'] = "_ts"

###

