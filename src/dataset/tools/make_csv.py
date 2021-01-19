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
    if dataset_path.exists():
        print("Found dataset!")
    
    examples = [f.absolute() for f in (dataset_path / "neighborhoods").glob("*.txt.gz")]
    print(f"Found {len(examples)}")

    # 1. Build a master record of all neighborhoods and examples
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
    
    # 2. Create the dataset
    # Eliminate any flights that just barely overlap - ensure adequate coverage. 
    if resample_flights:
        df_new = pd.DataFrame(columns=['examples', 'source_scan', 'target_scan', 'target_intensity'])
        print("Source scan examples by source")
        uf = {i: len(df[df.source_scan == i]) for i in df.source_scan.unique()}
        print(uf)
        print("total flights: ", len(uf))
        
        desired_count = 81000  # this is arbitrary, but works in this case

        for flight in uf:
            # take scans that are at least `desired_count` big
            if uf[flight] > desired_count:  
                df_flight = df.loc[(df['source_scan'] == flight)]
                df_new = df_new.append(df_flight)
     
        df = df_new
        print("Source scans after filtering")
        uf = {i: len(df[df.source_scan == i]) for i in df.source_scan.unique()}
        print(uf)
        print("total flights: ", len(uf))

    # Create training/validation/testing splits
    df = df.sample(frac=1).reset_index(drop=True)
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

    # Class Balancing
    # Target intensity class-balancing: the training set needs to be balanced 
    #     across the range of target intensities as well as for each source. 
    
    df_train_resampled = pd.DataFrame(columns=df_train.columns)
    sources = df_train.source_scan.unique()

    # Since there is a broad range, group intensities first, then balance by
    #     group. The following section defines group ranges, i.e., [0, 5), 
    #     [5, 10),..., [510, 515). 

    bin_size = 5
    bin_boundaries = [(i, i+bin_size) for i in range(0, 512, bin_size)]

    # A uniform sample per source is desired. The following code builds a 
    #     histogram of the examples over the target intensity. A desired
    #     sample size is then chosen. Dividing a bin by the number of sources
    #     yields the average bin size per source. This value will be used to 
    #     resample the intensity-groups per source. 

    hist, _ = np.histogram(df_train.target_intensity.values,
                               [i[1] for i in bin_boundaries])

    print("Bins: ", len(hist))

    desired_sample_size = int(np.median(hist))  # could use some other stat?
    sample_size_per_source = desired_sample_size // len(sources)

    # Perform resampling over each source for target intensity
    print("resampling over target intensities for each source scan")
    print(sample_size_per_source)

    for source in sources:
        # Get all examples for the current source
        curr_source_df = df_train.loc[df_train.source_scan == source]

        # Create a new dataframe to temporarily hold resampled data
        new_source_df = pd.DataFrame(columns=curr_source_df.columns)
    
        # resample source dataframe 
        for bin in bin_boundaries:
            l, h = bin

            # get all examples between these bounds
            examples = curr_source_df[(curr_source_df.target_intensity >= l) &
                                      (curr_source_df.target_intensity < h)]

            # resample examples and append. Somehow there are bins with zero
            #    examples, hence the if statement. This seems hard to believe.
            if len(examples):  
                new_source_df = new_source_df.append(
                    examples.sample(n=sample_size_per_source, replace=True),
                    ignore_index=True)


        # append balanced source to new dataframe
        df_train_resampled = df_train_resampled.append(
            new_source_df,
            ignore_index=True)
    
    # finished
    df_train = df_train_resampled

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
        
