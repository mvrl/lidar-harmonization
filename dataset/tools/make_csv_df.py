from pathlib import Path
import pandas as pd
import code

def make_csv_df(path):
    df_train = pd.read_csv(path / "train_dataset.csv", index_col=0)
    df_test = pd.read_csv(path / "test_dataset.csv", index_col=0)
    total_examples = 0
    my_flight = 37

    idx_train = []
    idx_test = []
    for i in range(len(df_train)):
        f = df_train.iloc[i]['flight_num']
        if f == my_flight:
            idx_train.append(i)
    
    for i in range(len(df_test)):
        f = df_test.iloc[i]['flight_num']  
        if f == my_flight:
            idx_test.append(i)

    df_train_df = df_train.iloc[idx_train]
    df_test_df = df_test.iloc[idx_test]

    df_train_df.to_csv(path / "train_dataset_df.csv")
    df_test_df.to_csv(path / "test_dataset_df.csv")
    print(f"Train examples: {len(df_train_df)}")
    print(f"Test examples: {len(df_test_df)}")
    print(f"created on {my_flight}")
    print("finished")

if __name__ == "__main__":
    make_csv_df(Path("150_190000"))
