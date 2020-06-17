import time
import code
import json
import numpy as np
from util.metrics import create_kde
print("got metrics")
from pathlib import Path
from pptk import kdtree, viewer
import matplotlib.pyplot as plt
from util.apply_rf import apply_rf


def load_laz(path):
    f = np.load(path)
    return f

def load_mapping(path):
    mapping = np.load(path)
    return mapping

def visualize_n_flights(path, flights):
    # flights is a list of numbers representing flights 0-40
    start_time = time.time()
    
    laz_files_path = Path(path)
    sample_size = 500000

    pts = np.load(laz_files_path / (str(flights[0])+".npy"))
    sample = np.random.choice(len(pts), size=sample_size)
    pts = pts[sample]
    
    for i in range(len(flights[1:])):
        fi = np.load(laz_files_path / (str(flights[i+1])+".npy"))
        sample = np.random.choice(len(fi), size=sample_size)
        pts = np.concatenate((pts, fi[sample])) 

    attr1 = pts[:, 3]
    attr2 = pts[:, 8]

    v = viewer(pts[:, :3])
    v.attributes(attr1, attr2)
    # code.interact(local=locals())


def create_big_tile_manual(path, base_flight, intersecting_flight, manual_point):

    laz_files_path = Path(path)
    json_file = open("dorfCurves.json")
    rf_data = json.load(json_file)
    mapping = load_mapping("mapping.npy")

    # plot the response function
    x = np.linspace(0, 1, 1000)
    m = mapping[intersecting_flight]
    plt.plot(
            x,
            np.interp(x,
                np.fromstring(rf_data[str(m)]['B'], sep=' '),
                np.fromstring(rf_data[str(m)]['I'], sep=' ')))

    plt.plot(x, x, 'k')
    plt.margins(x=0)
    plt.margins(y=0)
    plt.title("Response function on the big tile")
    plt.savefig("big_tile/response_plot_bt.png")
    print("Saved response plot")

    flight1 = np.load(laz_files_path / (str(base_flight)+".npy"))
    print(f"Loaded flight {flight1[0, 8]}")
    flight2 = np.load(laz_files_path / (str(intersecting_flight)+".npy"))
    print(f"Loaded flight {flight2[0, 8]}")
    
    kd1 = kdtree._build(flight1[:, :3])
    kd2 = kdtree._build(flight2[:, :3])

    q1 = kdtree._query(kd1, manual_point, k=1e6)
    q2 = kdtree._query(kd2, manual_point, k=5e5)

    tile_f1 = flight1[tuple(q1)]
    tile_f2 = flight2[tuple(q2)]
    tile = np.concatenate((tile_f1, tile_f2))

    # threshold the yaxis values to create a "clean" divide
    y_axis_mean = np.mean(tile, axis=0)[1]
    tile_f1 = tile_f1[np.logical_not(tile_f1[:, 1] <= y_axis_mean)]
    tile_f2 = tile_f2[np.logical_not(tile_f2[:, 1] > y_axis_mean)]
    tile = np.concatenate((tile_f1, tile_f2))
    # create corrupted tile_f2
    tile_f2_alt = apply_rf(
            "dorfCurves.json", 
            tile_f2, 
            mapping[intersecting_flight],
            512) 
   
    
    sample = np.random.choice(len(tile_f2), size=5000)
    create_kde(
            tile_f2[sample][:, 3],
            tile_f2_alt[sample][:, 3],
            "ground truth",
            "altered values",
            "big_tile/post_response_curve.png")
            
    print("Created post response curve kde")  # check
    
    attr1 = tile[:, 3]
    attr2 = tile[:, 8]
    attr3 = np.concatenate((tile_f1[:, 3], tile_f2_alt[:, 3]))
    
    v = viewer(tile[:, :3])
    fixed = Path("big_tile/big_tile_alt_fixed.npy")
    if fixed.exists():
        fixed_tile = np.load(fixed)
        attr4 = np.concatenate((tile_f1[:, 3], fixed_tile[:, 3]*512))
        print("Loaded fixed tile!")
        v.attributes(attr1, attr2, attr3, attr4)
        v.set(bg_color=(1,1,1,1))
        v.set(show_axis=False, show_grid=False, show_info=False)


    else:
        v.attributes(attr1, attr2, attr3)
        np.save("big_tile/big_tile_gt", tile_f2)
        np.save("big_tile/big_tile_alt", tile_f2_alt)

    code.interact(local=locals())



if __name__=='__main__':
    
    # Automatic big tile creation
    # create_big_tile('dublin_flights', 4, 1000000)
    
    # Create a big tile with manual point input
    create_big_tile_manual('dublin_flights/npy', 1, 37, 
       np.array([[316323.562, 234151.312, 12.875]]))
    
    exit()
    # Show the overlapping flights over flight 1
    visualize_n_flights(
            'dublin_flights/npy', 
             # [0, 1, 2, 4, 6, 7, 10, 15, 20, 21, 30, 35, 37, 39]))
             [1, 37])

    
                 
    
