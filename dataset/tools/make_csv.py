import numpy as np
import pandas as pd
from pathlib import Path


# Options
perform_check = False  # set to perform tileset verification, otherwise gen csv


alts = [f for f in Path(r"dataset/alt/").glob("*_alt.npy")]
gts = [f for f in Path(r"dataset/gt/").glob("*.npy")]
flight_path = []
flight_num = []
flight_files = [flight for flight in Path(r"dublaz").glob("*.laz")]

                
# sanity check that there is an altered version of each ground truth
for gt in gts:
    filename = gt.stem
    flight_num.append(filename.split("_")[0])
    alt_name = filename[:-4]+"_alt" + filename[-4:]
    altered = None

    if perform_check:
        # Check that our files are what we expect
        for alt in alts:
            if filename in str(alt):
                # reverse check as well to confirm
                alt_filename = alt.stem
                test = alt_filename[:-8]+alt_filename[-8:-4]
                if test in filename:
                    print(f"Found match: {alt} == {gt}") 
                    altered = alt
                    break

        if not altered:
           print("ERROR: couldn't find altered version of %s" % gt)
           exit()
        
        altered = np.load(altered)
        original = np.load(gt)
    
        # check that the xyz are the same and that intensities are different
        if not np.allclose(original[:, :3], altered[:, :3]):
            print("ERROR: xyz are different between %s and %s" % (gt, alt_name))
            exit()

        if np.allclose(original[:, 3], altered[:, 3]):
            print("ERROR: intensities were not altered between files")
            exit()

    flight_path.append(flight_files[int(filename[:filename.find("_")])])

print("SUCCESS")
print("creating csv...")
df = pd.DataFrame()
df['gt'] = gts
df['alt'] = alts
df['flight_num'] = flight_num
df['flight_path_file'] = flight_path

# randomize dataframe
df = df.sample(frac=1).reset_index(drop=True)

# create train/test split
sample_count = len(df)
split_point = sample_count - sample_count//5
print(f"split point: {split_point}")
df_train = df.iloc[:split_point, :]
df_test = df.iloc[split_point:, :]

df_train.to_csv("train_dataset.csv")
df_test.to_csv("test_dataset.csv")

print(f"Created training dataset with {len(df_train)} samples")
print(f"Created testing dataset with {len(df_test)} samples")

        
        
