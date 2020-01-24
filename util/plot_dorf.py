from pathlib import Path
import code
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D
# matplotlib.use('Agg')

# plot just the first 16...

def plot_dorf_functions(data):
    fig, axs = plt.subplots(4, 4)
    plt.tight_layout()
    plt.title("Camera Response Function Sample")
    fig.set_size_inches(9, 9)
    
    for i in range(16):
        axs.flat[i].plot(data['brightness'][str(i)],
                         data['intensities'][str(i)])
        axs.flat[i].title.set_text('CR %s' % i)
        axs.flat[i].set_xlabel('brightness')
        axs.flat[i].set_ylabel('intensity')
        
    plt.savefig("dataset/rf_plots.png")

# now plot the first point gt point cloud
def scatter3d(xyz, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=scalarMap.to_rgba(cs))
    fig.colorbar(scalarMap, label='intensity')
    plt.show()

def plot_dorf_cloud(data):
    files = [f for f in Path("dataset/gt/").glob('*.npy')]

    my_cloud = files[0]
    # center and delete origin
    my_cloud = np.load(my_cloud)
    my_cloud[:, :3] -=my_cloud[0][:3]

    scatter3d(my_cloud, my_cloud[:, 3])

    # perform alteration
    altered_cloud = np.copy(my_cloud)
    altered_cloud[:, 3] = np.interp(my_cloud[:, 3],
                                    np.array(data['brightness']['1'])*255,
                                    np.array(data['intensities']['1'])*255)

#    code.interact(local=locals())
    scatter3d(altered_cloud, altered_cloud[:, 3])
              


if __name__=='__main__':
    with open('dataset/response_functions.json') as json_file:
        data = json.load(json_file)

    plot_dorf_cloud(data)


    
    

    

