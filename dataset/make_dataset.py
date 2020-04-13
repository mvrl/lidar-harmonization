from tools.create_dataset import create_dataset
from tools.create_altered_tiles import create_altered_tiles, random_mapping
from tools.make_csv import make_csv

def make_dataset(neighborhood_size, sample_size, output_suffix=""):
    create_dataset('dublin_flights',
                   neighborhood_size,
                   sample_size,
                   #contains_flights=[4],
                   #output_suffix=output_suffix,
                   sanity_check=False)

    
    create_altered_tiles(f"{neighborhood_size}_{sample_size}{output_suffix}",
                         random_mapping())
    make_csv(f"{neighborhood_size}_{sample_size}{output_suffix}")

neighborhood_size = 0
sample_size = 10000
output_suffix = ""
print(f"Building dataset {neighborhood_size}_{sample_size}{output_suffix}")
make_dataset(neighborhood_size, sample_size, output_suffix)
