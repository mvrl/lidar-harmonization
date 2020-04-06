from tools.create_dataset import create_dataset
from tools.create_altered_tiles import create_altered_tiles
from tools.make_csv import make_csv

def make_dataset(neighborhood_size, sample_size):
    create_dataset('dublin_flights',
                   neighborhood_size,
                   sample_size,
                   # contains_flights=[4],
                   # output_prefix="single_flight",
                   sanity_check=False)

    create_altered_tiles(f"{neighborhood_size}_{sample_size}")
    make_csv(f"{neighborhood_size}_{sample_size}")

neighborhood_size = 150
sample_size = 100000
print(f"Building dataset {neighborhood_size}_{sample_size}")
make_dataset(neighborhood_size, sample_size)
