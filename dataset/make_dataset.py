import argparse
from tools.create_dataset import create_dataset
from tools.create_altered_tiles import create_altered_tiles, random_mapping
from tools.make_csv import make_csv

def make_dataset(neighborhood_size, example_count,  sample_size, output_suffix=""):
    create_dataset('dublin_flights',
                   neighborhood_size,
                   sample_size,
		   example_count,
                   # contains_flights=[7],
                   output_suffix=output_suffix,
                   sanity_check=False)

    
    # create_altered_tiles(f"{neighborhood_size}_{example_count}{output_suffix}",
#                         random_mapping())

    make_csv(f"{neighborhood_size}_{example_count}{output_suffix}", example_count)


parser = argparse.ArgumentParser()
parser.add_argument("neighborhood_size")
parser.add_argument("sample_size")
parser.add_argument("required_examples")
parser.add_argument("output_suffix")

args = vars(parser.parse_args())

print(args)

neighborhood_size = int(args["neighborhood_size"])
sample_size = int(args["sample_size"])
required_examples = int(args["required_examples"]) # will be divided between test, train, and validation
output_suffix = args["output_suffix"]

print(f"Building dataset {neighborhood_size}_{required_examples}{output_suffix}")
make_dataset(neighborhood_size, required_examples, sample_size, output_suffix)
