from tools.create_dataset import create_dataset
from tools.create_altered_tiles import create_altered_tiles, random_mapping
from tools.make_csv import make_csv

def make_dataset(neighborhood_size, example_count,  sample_size, output_suffix=""):
    create_dataset('dublin_flights',
                   neighborhood_size,
                   sample_size,
		   example_count,
                   contains_flights=[7],
                   output_suffix=output_suffix,
                   sanity_check=False)

    
    create_altered_tiles(f"{neighborhood_size}_{example_count}{output_suffix}",
                         random_mapping())
    make_csv(f"{neighborhood_size}_{example_count}{output_suffix}")

neighborhood_size = 10
sample_size = 100000
required_examples = 8000  # will be divided between test, train, and validation
output_suffix = "_df7" 
print(f"Building dataset {neighborhood_size}_{required_examples}{output_suffix}")
make_dataset(neighborhood_size, required_examples, sample_size, output_suffix)
