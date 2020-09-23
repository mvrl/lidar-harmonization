import code
from pathlib import Path
import pandas as pd

def combine_datasets(hz_ds, intp_ds, sample_ratio=1):
    # combine datasets into one super-dataset
    save_path = Path("combined")
    save_path.mkdir(parents=True, exist_ok=True)

    hz_ds = Path(hz_ds)
    intp_ds = Path(intp_ds)
    
    # Get paths for each dataset
    hz_train = hz_ds / "train_dataset.csv"
    intp_train = intp_ds / "train_dataset.csv"

    hz_val = hz_ds / "val_dataset.csv"
    intp_val = intp_ds / "val_dataset.csv"

    hz_test = hz_ds / "test_dataset.csv"
    intp_test = intp_ds / "test_dataset.csv"

    # Load and combine
    hz_train_df = pd.read_csv(hz_train, index_col=0)
    intp_train_df = pd.read_csv(intp_train, index_col=0) 
    
    hz_val_df = pd.read_csv(hz_val, index_col=0)
    intp_val_df = pd.read_csv(intp_val, index_col=0)

    hz_test_df = pd.read_csv(hz_test, index_col=0)
    intp_test_df = pd.read_csv(intp_test, index_col=0)

    if sample_ratio < 1:
        print(f"resampling datasets with ratio {sample_ratio}")
        hz_train_df = hz_train_df.sample(frac=sample_ratio)
        intp_train_df = intp_train_df.sample(frac=sample_ratio)

        hz_val_df = hz_val_df.sample(frac=sample_ratio)
        intp_val_df = intp_val_df.sample(frac=sample_ratio)

        hz_test_df = hz_test_df.sample(frac=sample_ratio)
        intp_test_df = intp_test_df.sample(frac=sample_ratio)

    combine_train_df = pd.concat((hz_train_df, intp_train_df))
    combine_val_df = pd.concat((hz_val_df, intp_val_df))
    combine_test_df = pd.concat((hz_test_df, intp_test_df))
    
    # Save
    combine_train_df.to_csv(save_path / "train_dataset.csv")
    combine_val_df.to_csv(save_path / "val_dataset.csv")
    combine_test_df.to_csv(save_path / "test_dataset.csv")

if __name__=="__main__":
    combine_datasets("150_190000", "interpolation_dataset", sample_ratio=0.1)
