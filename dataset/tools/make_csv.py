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
    intensities = [None] * len(gts)
    flight_num = [None] * len(gts)
    flight_files = [flight for flight in Path(r"dublin_flights/npy").glob("*.npy")]

    gt_ordered_list = [None] * len(gts)
    alt_ordered_list = [None] * len(gts)
    # sanity check that there is an altered version of each ground truth
    for i in trange(len(gts), desc="processing", ascii=True):
        filename = gts[i].stem
        flight_num[i] = filename.split("_")[0]
        target_intensity = filename.split("_")[1]
        
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
        intensities[i] = target_intensity
    

    print("SUCCESS")
    print("creating csv...")
    df = pd.DataFrame()
    df['gt'] = gt_ordered_list
    df['alt'] = alt_ordered_list
    df['flight_num'] = flight_num
    df['target_intensity'] = intensities
    df['flight_path_file'] = flight_path

    df[['flight_num', 'target_intensity']] = df[['flight_num', 'target_intensity']].apply(pd.to_numeric)
    print("Saving dataset")
    df.to_csv(dataset_path / "master.csv")
    
    # Sample by flight number first
    df_new = pd.DataFrame(columns=['gt', 'alt', 'flight_num', 'target_intensity', 'flight_path_file'])
    uf = {i: len(df[df.flight_num == i]) for i in df.flight_num.unique()}
    min_flight = uf[min(uf, key=uf.get)]
    
    for flight in uf:
        df_flight = df.loc[df['flight_num'] == flight]
        df_new = df_new.append(df_flight.sample(n=min_flight), ignore_index=True, sort=False)
     
    df = df_new
    
    
    df = df.sample(frac=1).reset_index(drop=True) # randomize rows
    print(df.columns)

    # Note: oversampling only works AFTER you create train/test splits, otherwise 
    #       evaluation samples will bleed into the training set. This means a
    #       training-validation set must also be built. 
    # Create new dataframe with samples evenly distributed for target intensities
    # create train/test split
    sample_count = len(df)
    split_point = sample_count - sample_count//5
    df_train = df.iloc[:split_point, :].reset_index(drop=True)
    df_test = df.iloc[split_point:, :].reset_index(drop=True)

    val_split_point = len(df_train) - len(df_train)//5
    df_val = df_train.iloc[val_split_point:, :].reset_index(drop=True)
    df_train = df_train.iloc[:val_split_point, :].reset_index(drop=True)
    print(f"Training samples pre-oversampling: {len(df_train)}")
    print(f"Validation samples pre-oversampling: {len(df_val)}")
    print(f"Testing samples pre-oversampling: {len(df_test)}")

    # Use bins of 5 to balance out the intensities
    bin_sizes = []
    bin_size = 10
    df_new = pd.DataFrame(columns=['gt', 'alt', 'flight_num', 'target_intensity', 'flight_path_file'])
    bin_boundaries = [(i, i+bin_size) for i in range(5, 515, bin_size)]

    # Calcualte bin sizes
    for (l, h) in bin_boundaries:
        if l >= 505:
            continue
        else:
            bin_sizes.append(len(df_train[(df_train.target_intensity >= l) & (df_train.target_intensity < h)]))
    bin_sizes = np.array(bin_sizes)
    
    # sample size as maximum bin size? Perhaps it would be best to undersample the top bin since it is not 
    # representative of what that bin actually is. 
    sample_size = bin_sizes.max()
    print("Sample Counts:")
    for (l, h) in bin_boundaries:
        new_df =  df_train[(df_train.target_intensity >= l) & (df_train.target_intensity < h)]
        new_sample = new_df.sample(n=sample_size, replace=True)
        df_new = df_new.append(new_sample, ignore_index=True, sort=False)
        print(f"{l}-{h}: {len(new_df)} | {len(new_sample)}")
    
    
    code.interact(local=locals())
    df_new = df_new.loc[:, 'gt':'flight_num']    
    df_train = df_new
    df_test = df_test.loc[:, 'gt':'flight_num']
    df_val = df_val.loc[:, 'gt':'flight_num']
    df_train.to_csv(dataset_path / "train_dataset.csv")
    df_val.to_csv(dataset_path / "val_dataset.csv")
    df_test.to_csv(dataset_path / "test_dataset.csv")

    print(f"Created training dataset with {len(df_train)} samples")
    print(f"Created validation dataset with {len(df_val)} samples")
    print(f"Created testing dataset with {len(df_test)} samples")

if __name__=='__main__':
    make_csv("150_190000", 8000)

        
