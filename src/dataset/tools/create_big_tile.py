import time
import code
import json
import numpy as np
from src.dataset.tools.metrics import create_kde
print("got metrics")
from pathlib import Path
from pptk import kdtree, viewer
import matplotlib.pyplot as plt
from src.dataset.tools.apply_rf import apply_rf


def load_laz(path):
    f = np.load(path)
    return f

def load_mapping(path):
    mapping = np.load(path)
    return mapping

def visualize_n_flights(path, flights, sample_size=500000):
    # flights is a list of numbers representing flights 0-40
    start_time = time.time()
    
    laz_files_path = Path(path)
   
    pts = np.load(laz_files_path / (str(flights[0])+".npy"))
    sample = np.random.choice(len(pts), size=sample_size)
    if sample_size:
        pts = pts[sample]
    
    
    for i in range(len(flights[1:])):
        
        fi = np.load(laz_files_path / (str(flights[i+1])+".npy"))
        if sample_size:
            sample = np.random.choice(len(fi), size=sample_size)
            pts = np.concatenate((pts, fi[sample])) 
        else:
            pts = np.concatenate((pts, fi))

    attr1 = pts[:, 3]
    attr2 = pts[:, 8]

    v = viewer(pts[:, :3])
    v.attributes(attr1, attr2)
    # code.interact(local=locals())


def create_big_tile_manual(path, base_flight, intersecting_flight, manual_point, num_points=1e6, in_overlap=True):

    laz_files_path = Path(path)
    json_file = open("dorf.json")
    rf_data = json.load(json_file)
    mapping = load_mapping("mapping.npy")
    if in_overlap:
        save_dir = Path("big_tile_in_overlap")
    else:
        save_dir = Path("big_tile_no_overlap")

    save_dir.mkdir(parents=True, exist_ok=True)

    # plot the response function - this plot should show the transformation
    # that is applied to the source flight in the generated tile. Compare this
    # with post_response_curve generated at the end. 

    x = np.linspace(0, 1, 1000)
    m = mapping[intersecting_flight]
    plt.plot(
            x,
            np.interp(
                x,
                np.fromstring(rf_data[str(m)]['B'], sep=' '),
                np.fromstring(rf_data[str(m)]['I'], sep=' ')))

    plt.plot(x, x, 'k')
    plt.margins(x=0)
    plt.margins(y=0)
    plt.title("Response function on the big tile")
    plt.savefig(save_dir / "pre_response_curve.png")
    print("Saved response plot")
    
    # if we are in the overlap, we have to process two kd trees, 
    # otherwise we can just do 1. 
    if in_overlap:
        flight1 = np.load(laz_files_path / (str(base_flight)+".npy"))
        print(f"Loaded flight {flight1[0, 8]}")
        kd1 = kdtree._build(flight1[:, :3])
        q1 = kdtree._query(kd1, manual_point, k=1e6)
        tile_f1 = flight1[tuple(q1)]
    flight2 = np.load(laz_files_path / (str(intersecting_flight)+".npy"))
    print(f"Loaded flight {flight2[0, 8]}")
    
    kd2 = kdtree._build(flight2[:, :3])
    q2 = kdtree._query(kd2, manual_point, k=1e6)
    tile_f2 = flight2[tuple(q2)]

    if in_overlap:
        tile_f2_1 = tile_f2.copy()
        tile_f1_2 = tile_f1.copy()
        tile_f2_1[:, 8] = float(base_flight)
        tile_f1_2[:, 8] = float(intersecting_flight)
     
        # threshold the axis values to create a "clean" divide
        # 0 = x axis, 1 = y axis, 2 = z axis (??)
        my_axis=0
        tile_f1 = tile_f1[np.logical_not(tile_f1[:, my_axis] <= manual_point[0, my_axis])]
        tile_f2 = tile_f2[np.logical_not(tile_f2[:, my_axis] > manual_point[0, my_axis])]

        # Fill in missing points from overlap
        tile_f2_1 = tile_f2_1[np.logical_not(tile_f2_1[:, my_axis] <= manual_point[0, my_axis])]
        tile_f1_2 = tile_f1_2[np.logical_not(tile_f1_2[:, my_axis] > manual_point[0, my_axis])]
        tile_f1 = np.concatenate((tile_f1, tile_f2_1))
        tile_f2 = np.concatenate((tile_f2, tile_f1_2))

    # create corrupted tile_f2
    tile_f2_alt = apply_rf(
            "dorf.json", 
            tile_f2, 
            mapping[intersecting_flight],
            512) 
   
    sample = np.random.choice(len(tile_f2), size=5000)
    create_kde(
            tile_f2[sample][:, 3],
            tile_f2_alt[sample][:, 3],
            "ground truth",
            "altered values",
            save_dir / "post_response_curve.png")
            
    print("Created post response curve kde")
    
    if in_overlap:
        tile_gt = np.concatenate((tile_f1, tile_f2))
        tile_alt = np.concatenate((tile_f1, tile_f2_alt))
        attr1 = tile_gt[:, 3]
        attr2 = tile_alt[:, 8]
        attr3 = tile_alt[:, 3]
        v = viewer(tile_gt[:, :3])

    else:
        attr1 = tile_f2[:, 3]
        attr2 = tile_f2[:, 8]
        attr3 = tile_f2_alt[:, 3]
        v = viewer(tile_f2[:, :3])
    
    v.attributes(attr1, attr2, attr3)
    v.set(lookat=manual_point[0])

    if in_overlap:

        np.save(save_dir / "base_flight_tile.npy", tile_f1)

    np.save(save_dir / "gt.npy", tile_f2)
    np.save(save_dir / "alt.npy", tile_f2_alt)


    code.interact(local=locals())

if __name__=='__main__':
    
    # Automatic big tile creation
    # create_big_tile('dublin_flights', 4, 1000000)
    
    # Create a big tile with manual point input in the overlap region
    create_big_tile_manual('dublin/npy', 1, 37, 
          np.array([[316340.656, 234122.562, 6.368]]))
    
    # exit()
    # create a big tile with manual point input outside the overlap region
    # create_big_tile_manual('dublin/npy', 1, 37,
    #         np.array([[316557.094, 234186.656, 1.191]]), in_overlap=False)
    # Show the overlapping flights over flight 1
    # visualize_n_flights(
    #       'dublin/npy', 
    #        [0, 1, 2, 4, 6, 7, 10, 15, 20, 21, 30, 35, 37, 39])
    #        [1, 37], sample_size=None)

    
                 
    
