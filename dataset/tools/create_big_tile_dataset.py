from code import interact
import numpy as np
import pandas as pd
from pathlib import Path
from pptk import kdtree
from util.metrics import create_kde

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

    # get neighborhoods
    for i in range(len(query)):
        gt = big_tile_gt[query[i]]
        alt = big_tile_alt[query[i]]
        
        np.save(gt_path / f"{i}.npy", gt)
        np.save(alt_path / f"{i}.npy", alt)

    # create csv
    gt_files = [f.absolute() for f in gt_path.glob("*.npy")]
    alt_files = [f.absolute() for f in alt_path.glob("*.npy")]
    
    gt_files.sort()
    alt_files.sort()

    df = pd.DataFrame()
    df["gt"] = gt_files
    df["alt"] = alt_files

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



