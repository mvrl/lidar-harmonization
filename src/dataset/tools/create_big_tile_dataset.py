import code
from src.dataset.tools.metrics import create_kde
print("got metrics!")
import numpy as np
import pandas as pd
from pathlib import Path
from pptk import kdtree
from tqdm import tqdm, trange

def create_big_tile_dataset(path, neighborhood_size=150):
    path = Path(path)
    save_path = path / "neighborhoods"
    save_path.mkdir(parents=True, exist_ok=True) 
    big_tile_gt = np.load(path / "gt.npy")
    big_tile_alt = np.load(path / "alt.npy")

    # Confirm that these tiles are equivalent except in the 
    # intensity channel
    if not (np.allclose(big_tile_gt[:, :3], big_tile_alt[:, :3])
            and np.allclose(big_tile_gt[:, 4:], big_tile_alt[:, 4:]) 
            and not np.allclose(big_tile_gt[:, 3], big_tile_alt[:, 3])):
        exit("Error! Tiles are no equivalent or the intensities are the same.")

    # measure the response curve again, check with pre and post
    # from tile creation.
    sample = np.random.choice(len(big_tile_gt[:, 3]), size=5000)
    create_kde(
            big_tile_gt[sample][:, 3],
            big_tile_alt[sample][:, 3],
            "ground truth intensity",
            "altered intensity",
            path / "measured_response_curve.png",
            sample_size=5000)
    print("generated response curve")
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
    examples = [None] * len(query)
    fid = [None] * len(query)
    intensities = [None] * len(query)

    # get neighborhoods
    for i in trange(len(query), desc="querying neighborhoods", ascii=True):
        gt_query = big_tile_gt[query[i]]
        alt_query = big_tile_alt[query[i]]
        
        # Sanity check that these are the same
        if (np.allclose(gt_query[:, :3], alt_query[:, :3]) != True and
                np.allclose(gt_query[:, 4:], alt_query[:, 4:]) != True and
                np.allclose(gt_query[:, 3], alt_query[:, 3] != False)):
            exit("mismatch elements!")

        # Keep parity with training dataset - save example as (152, 9) point cloud
        # there will be an extra copy of the center point altered ground truth at idx 1
        # that will be thrown out by the model.
        my_example = np.concatenate((
            np.expand_dims(gt_query[0, :], 0), 
            np.expand_dims(alt_query[0, :], 0),
            alt_query))
        
        np.save(save_path / f"{i}.npy", my_example)
        examples[i] = (save_path / f"{i}.npy").absolute()
        fid[i] = my_example[0, 8]  # flight number
        intensities[i] = int(my_example[0, 4])


    # create csv
    df = pd.DataFrame()
    df["examples"] = examples
    df["flight_num"] = fid
    # df["target_intensity"] = intensities

    df.to_csv(path / "big_tile_dataset.csv")


if __name__ == "__main__":
    print('starting...')
    create_big_tile_dataset(r"big_tile")



