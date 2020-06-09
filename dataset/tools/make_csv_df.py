from pathlib import Path
import pandas as pd
import code

def make_csv_df(path):
    df_train = pd.read_csv(path / "train_dataset.csv", index_col=0)
    df_test = pd.read_csv(path / "test_dataset.csv", index_col=0)
    flights = {}
    total_examples = 0
    for i in range(len(df_train)):
        f = df_train['gt'][i].split("_")[1].split("/")[-1]
        if f in flights:
            flights[f] += 1
            total_examples+=1
        else:
            flights[f] = 1
            total_examples += 1 

    print(flights)
    max_flight = max(flights, key=flights.get)

    idx_max_train = []
    idx_max_test = []
    for i in range(len(df_train)):
        f = df_train['gt'].iloc[i].split("_")[1].split("/")[-1]
        if f == max_flight:
            idx_max_train.append(i)
    
    for i in range(len(df_test)):
        f = df_test['gt'].iloc[i].split("_")[1].split("/")[-1]  
        if f == max_flight:
            idx_max_test.append(i)

    df_train_df = df_train.iloc[idx_max_train]
    df_test_df = df_test.iloc[idx_max_test]

    df_train_df.to_csv(path / "train_dataset_df.csv")
    df_test_df.to_csv(path / "test_dataset_df.csv")
    print(total_examples)
    print("finished")

if __name__ == "__main__":
    make_csv_df(Path("200_78000"))
