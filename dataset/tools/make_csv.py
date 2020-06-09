import numpy as np
import pandas as pd
from pathlib import Path
import code
from tqdm import tqdm, trange
# Options
# N/A


def make_csv(path, example_count):
    print(f"building csv on {path}")
    dataset_path = Path(path)

    alts = [f.absolute() for f in (dataset_path / "alt").glob("*.npy")]
    gts = [f.absolute() for f in (dataset_path / "gt").glob("*.npy")]
    
    flight_path = [None] * len(gts)
    flight_num = [None] * len(gts)
    flight_files = [flight for flight in Path(r"dublin_flights").glob("*.npy")]

    gt_ordered_list = [None] * len(gts)
    alt_ordered_list = [None] * len(gts)
    # sanity check that there is an altered version of each ground truth
    for i in trange(len(gts), desc="processing", ascii=True):
        filename = gts[i].stem
        flight_num[i] = filename.split("_")[0]
        alt_name = None
        altered = None

        # Check that our files are what we expect
        alt_name = alts[i].stem
        if alt_name != filename:
            exit("ERROR: wrong filename")
        else:
            altered = alts[i]

        flight_path[i] = (flight_files[int(filename[:filename.find("_")])].absolute())
        gt_ordered_list[i] = gts[i]
        alt_ordered_list[i] = altered
    

    print("SUCCESS")
    print("creating csv...")
    df = pd.DataFrame()
    df['gt'] = gt_ordered_list
    df['alt'] = alt_ordered_list
    df['flight_num'] = flight_num
    df['flight_path_file'] = flight_path

    # Create new dataframe with the ~required number of samples, 
    # evenly sampled from each flight
    #
    df_new = pd.DataFrame(columns=['gt', 'alt'])
    
    # get a list of unique flights from the dataframe
    uf = (df.flight_num.unique())

    # calculate how many samples exist for each flight 
    remaining_samples = {f:len(df.flight_num[df.flight_num == f]) for f in uf}
    
    # estimate the number of samples needed from each flight
    z = int(example_count//len(remaining_samples))

    # if a flight doesn't have enough samples, remove it
    viable_flights = {key: value//z for key,value in remaining_samples.items()}
    keys_to_remove = []
    for key,value in viable_flights.items():
        if value == 0:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del viable_flights[key]
    
    viable_flights = list(viable_flights.keys())
    
    # sample from the list of viable flights,
    # create a new dataframe to hold this data
    for flight in viable_flights:
        tdf = df[df.flight_num == flight]
        df_new = df_new.append(tdf.sample(n=z).loc[:, 'gt':'alt'], ignore_index=True)
    code.interact(local=locals()) 
    df = df_new
    df = df.sample(frac=1).reset_index(drop=True)
  
    # create train/test split
    sample_count = len(df)
    split_point = sample_count - sample_count//5
    print(f"split point: {split_point}")
    df_train = df.iloc[:split_point, :]
    df_test = df.iloc[split_point:, :]

    df_train.to_csv(dataset_path / "train_dataset.csv")
    df_test.to_csv(dataset_path / "test_dataset.csv")

    print(f"Created training dataset with {len(df_train)} samples")
    print(f"Created testing dataset with {len(df_test)} samples")

if __name__=='__main__':
    make_csv("200_78000", 78000)

        
