import numpy as np
import pandas as pd
from pathlib import Path
import code
from tqdm import tqdm, trange
# Options
# N/A


def make_csv(path):
    dataset_path = Path(path)

    alts = [f.absolute() for f in (dataset_path / "alt").glob("*.npy")]
    gts = [f.absolute() for f in (dataset_path / "gt").glob("*.npy")]
    
    flight_path = []
    flight_num = []
    flight_files = [flight for flight in Path(r"dublin_flights").glob("*.npy")]

    gt_ordered_list = []
    alt_ordered_list = []
    # sanity check that there is an altered version of each ground truth
    for i in trange(len(gts), desc="processing", ascii=True):
        filename = gts[i].stem
        flight_num.append(filename.split("_")[0])
        alt_name = None
        altered = None

        # Check that our files are what we expect
        for alt in alts:
            alt_name = alt.stem
                # reverse check as well to confirm
            if alt_name == filename:
                # print(f"Found match: {alt} == {gts[i]}") 
                altered = alt
                break

        if not altered:
            print("ERROR: couldn't find altered version of %s" % gts[i])
            exit()

        altered_np = np.load(altered)
        original_np = np.load(gts[i])

        # check that the xyz are the same and that intensities are different
        if not np.allclose(original_np[:, :3], altered_np[:, :3]):
            print("ERROR: xyz are different between %s and %s" % (gts[i], alt_name))
            exit()

        if np.allclose(original_np[:, 3], altered_np[:, 3]):
            if not (np.all(original_np[:, 3]) and original_np[:, 3][0] == 512):
                code.interact(local=locals())
                print("ERROR: intensities were not altered between files")
                exit()

        flight_path.append(flight_files[int(filename[:filename.find("_")])].absolute())
        gt_ordered_list.append(gts[i])
        alt_ordered_list.append(altered)


    print("SUCCESS")
    print("creating csv...")
    df = pd.DataFrame()
    df['gt'] = gt_ordered_list
    df['alt'] = alt_ordered_list
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

    df_train.to_csv(dataset_path / "train_dataset.csv")
    df_test.to_csv(dataset_path / "test_dataset.csv")

    print(f"Created training dataset with {len(df_train)} samples")
    print(f"Created testing dataset with {len(df_test)} samples")

if __name__=='__main__':
    make_csv("50_10000")
        
