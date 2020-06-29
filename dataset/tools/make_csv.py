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

    examples = [f.absolute() for f in (dataset_path / "neighborhoods").glob("*.npy")]
    intensities = [None] * len(examples)
    flight_num = [None] * len(examples)
    

    for i in trange(len(examples), desc="processing", ascii=True):
        filename = examples[i].stem
        flight_num[i] = filename.split("_")[0]
        target_intensity = filename.split("_")[1]
        intensities[i] = target_intensity
    

    print("creating csv...")
    df = pd.DataFrame()
    df['examples'] = examples
    df['flight_num'] = flight_num
    df['target_intensity'] = intensities

    df[['flight_num', 'target_intensity']] = df[['flight_num', 'target_intensity']].apply(pd.to_numeric)
    print("Saving dataset")
    df.to_csv(dataset_path / "master.csv")
    
    # Sample by flight number first
    df_new = pd.DataFrame(columns=['examples', 'flight_num', 'target_intensity'])
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
    df_new = pd.DataFrame(columns=['examples', 'flight_num', 'target_intensity', 'flight_path_file'])
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
    
    
    df_train = df_new.loc[:, 'examples':'flight_num']    
    df_val = df_val.loc[:, 'examples':'flight_num']
    df_test = df_test.loc[:, 'examples':'flight_num']

    # sanity check that no values from df_train exist in df_val or df_test
    for my_df in (df_val, df_test):
        test = pd.merge(my_df, df_train)
        if len(test) > 0:
            print(f"SANITY CHECK FAILED: there is overlap from training in testing/validation")
            code.interact(local=locals())

    print("Sanity Check: Success!")

    df_train.to_csv(dataset_path / "train_dataset.csv")
    df_val.to_csv(dataset_path / "val_dataset.csv")
    df_test.to_csv(dataset_path / "test_dataset.csv")

    print(f"Created training dataset with {len(df_train)} samples")
    print(f"Created validation dataset with {len(df_val)} samples")
    print(f"Created testing dataset with {len(df_test)} samples")

if __name__=='__main__':
    make_csv("150_190000", 8000)

        
