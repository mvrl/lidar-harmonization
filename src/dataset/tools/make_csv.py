import numpy as np
import pandas as pd
from pathlib import Path
import code
from tqdm import tqdm, trange
# Options
# N/A


def make_csv(path, resample_flights=True):
    print(f"building csv on {path}")
    dataset_path = Path(path)
    print(dataset_path.exists())

    examples = [f.absolute() for f in (dataset_path / "neighborhoods").glob("*.txt.gz")]
    print(f"Found {len(examples)}")
    intensities = [None] * len(examples)
    source_scan = [None] * len(examples)
    target_scan = [None] * len(examples)
    

    for i in trange(len(examples), desc="processing", ascii=True):
        filename = examples[i].stem
        source_scan[i] = filename.split("_")[0]
        target_scan[i] = filename.split("_")[1]
        intensities[i] = filename.split("_")[2]
    

    print("creating csv...")
    df = pd.DataFrame()
    df['examples'] = examples
    df['source_scan'] = source_scan
    df['target_scan'] = target_scan
    df['target_intensity'] = intensities

    df[['source_scan', 'target_scan', 'target_intensity']] = df[['source_scan', 'target_scan', 'target_intensity']].apply(pd.to_numeric)
    print("Saving dataset")
    df.to_csv(dataset_path / "master.csv")
    
    # Undersample by source scan sample size
    if resample_flights:
        df_new = pd.DataFrame(columns=['examples', 'source_scan', 'target_scan', 'target_intensity'])
        print("Source scan examples by source")
        uf = {i: len(df[df.source_scan == i]) for i in df.source_scan.unique()}
        print(type(target_scan))
        print(uf)
        print(len(uf))
        # min_flight = uf[min(uf, key=uf.get)]
    
        for flight in uf:
            # take only the largest scans!
            if uf[flight] > 81000:
                df_flight = df.loc[(df['source_scan'] == flight)]
                df_new = df_new.append(
                    df_flight.sample(n=81000), 
                    ignore_index=True, 
                    sort=False)
     
        df = df_new
        print("Source scan examples by source *after* resampling -- check this!")
        uf = {i: len(df[df.source_scan == i]) for i in df.source_scan.unique()}
        print(uf)
        print(len(uf))
        input("Press CTRL-C now if this seems wrong, else ENTER")
    df = df.sample(frac=1).reset_index(drop=True) # randomize rows
    print(df.columns)

    # Note: oversampling only works after train/val/test splits, otherwise 
    #       evaluation samples will bleed into the training set.

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
    bin_size = 5
    df_new = pd.DataFrame(columns=['examples', 'source_scan', 'target_scan', 'target_intensity'])
    bin_boundaries = [(i, i+bin_size) for i in range(0, 512, bin_size)]

    # Calculate bin sizes
    for (l, h) in bin_boundaries:
            bin_sizes.append(len(df_train[(df_train.target_intensity >= l) & (df_train.target_intensity < h)]))
    bin_sizes = np.array(bin_sizes)
    
    # the intensities are clipped at 512, so there are a 
    # disproportionate number of examples in the last bin
    desired_bin_size = int(bin_sizes[:-1].mean())

    for i, bin_size in enumerate(bin_sizes):
        new_df = df_train[
            (df_train.target_intensity >= bin_boundaries[i][0]) & 
            (df_train.target_intensity  < bin_boundaries[i][1])]
        if len(new_df) <= 0:
            continue
        r = True if bin_size < desired_bin_size else False
        new_sample = new_df.sample(n=desired_bin_size, replace=r)
        df_new = df_new.append(new_sample, ignore_index=True, sort=False)

    df_train = df_new

    # sanity check that no values from df_train exist in df_val or df_test
    for my_df in (df_val, df_test):
        test = pd.merge(my_df, df_train)
        if len(test) > 0:
            print(f"SANITY CHECK FAILED: there is overlap from training in testing/validation")
            code.interact(local=locals())

    print("Sanity Check: Success!")

    df_train.to_csv(dataset_path / "train.csv")
    df_val.to_csv(dataset_path / "val.csv")
    df_test.to_csv(dataset_path / "test.csv")

    print(f"Created training dataset with {len(df_train)} samples")
    print(f"Created validation dataset with {len(df_val)} samples")
    print(f"Created testing dataset with {len(df_test)} samples")

if __name__=='__main__':

    make_csv("synth_crptn/150", resample_flights=True)
    make_csv("synth_crptn+shift/150", resample_flights=True)
        
