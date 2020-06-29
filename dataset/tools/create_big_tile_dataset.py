import code
import numpy as np
import pandas as pd
from pathlib import Path
from pptk import kdtree
from util.metrics import create_kde
from tqdm import tqdm, trange

def create_big_tile_dataset(path, neighborhood_size=150):
    path = Path(path)
    save_path = path / "neighborhoods"
    save_path.mkdir(parents=True, exist_ok=True)
    gt_path = save_path / "gt"
    gt_path.mkdir(parents=True, exist_ok=True)
    alt_path = save_path / "alt"
    alt_path.mkdir(parents=True, exist_ok=True)

    big_tile_alt = np.load(path / "big_tile_alt.npy")
    big_tile_gt = np.load(path / "big_tile_gt.npy")

    sample = np.random.choice(len(big_tile_gt[:, 3]), size=5000)
    create_kde(
            big_tile_gt[sample][:, 3],
            big_tile_alt[sample][:, 3],
            "ground truth intensity",
            "altered intensity",
            path / "measured_response_curve.png")

    kd = kdtree._build(big_tile_alt[:, :3])

    query = kdtree._query(
            kd, 
            big_tile_alt[:, :3],
            k=neighborhood_size)

    my_query = []
    for i in query:
        if len(i) == neighborhood_size:
            my_query.append(i)

    good_sample_ratio = ((len(query) - len(my_query))/len(query)) * 100
    print(f"Found {good_sample_ratio} perecent of points with not enough close neighbors!")

    query = my_query
    gt = [None] * len(query)
    alt = [None] * len(query)
    fid = [None] * len(query)

    # get neighborhoods
    for i in trange(len(query), desc="querying neighborhoods", ascii=True):
        gt_query = big_tile_gt[query[i]]
        alt_query = big_tile_alt[query[i]]
        
        # Sanity check that these are the same
        if (np.allclose(gt_query[:, :3], alt_query[:, :3]) != True and
                np.allclose(gt_query[:, 4:], alt_query[:, 4:]) != True and
                np.allclose(gt_query[:, 3], alt_query[:, 3] != False)):
            exit("mismatch elements!")
        
        np.save(gt_path / f"{i}.npy", gt_query)
        gt[i] = (gt_path / f"{i}.npy").absolute()
        np.save(alt_path / f"{i}.npy", alt_query)
        alt[i] = (alt_path / f"{i}.npy").absolute()
        fid[i] = gt_query[0, 8]  # flight number


    # create csv
    df = pd.DataFrame()
    df["gt"] = gt
    df["alt"] = alt
    df["flight_num"] = fid

    # Sanity check!
    for idx, row in df.iterrows():
        if Path(row["gt"]).stem != Path(row["alt"]).stem:
            print("mismatch rows!")
            exit()
            break

    print("no errors")

    df.to_csv(path / "big_tile_dataset.csv")


if __name__ == "__main__":
    create_big_tile_dataset(r"big_tile")



