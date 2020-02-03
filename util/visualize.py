import code
import numpy as np
from pptk import viewer
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cmx
from mpl_toolkits.mplot3d import Axes3D

# some different visualization methods
# cs = colorscale, just the intensity channel can be supplied here

def scatter3d(xyz, cs, colorsMap='jet'):
    cm = plt.get_cmap(colorsMap)
    cNorm = matplotlib.colors.Normalize(vmin=min(cs), vmax=max(cs))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=scalarMap.to_rgba(cs))
    fig.colorbar(scalarMap, label='intensity')
    plt.show()


def pptk_view(cloud, point_size=0.02, show_axis=False, bg_color=[1,1,1,1], show_grid=False):
    v = viewer(cloud[:, :3])
    v.attributes(cloud[:,3])
    v.set(point_size=point_size,
          show_axis=show_axis,
          bg_color=bg_color,
          show_grid=show_grid)

    return v

def pptk_view3(clouds, offset=2, point_size=0.02, show_axis=False, bg_color=[1,1,1,1], show_grid=False):
    # assumes clouds are loaded in as (gt, alt, fixed), color codes for identification
    if len(clouds) != 3:
        print(f"Incorrect number of clouds, found {len(clouds)}, expected 3")
        return
    
    gt, alt, fixed = clouds
    code.interact(local=locals())
    disp = np.concatenate((
        fixed,
        alt[:,1]+offset,
        gt[:,1]-offset), axis=0)

    print("starting viewer")

    attribute_lengh = len(gt[:,3])
    gt_color = np.full(attribute_lengh, 0)
    alt_color = np.fill(attribute_lengh, 127)
    fixed_color = np.fill(attribute_lengh, 255)
    disp_color = np.concatenate((fixed_color, alt_color, gt_color))
    
    v = viewer(disp[:, :3])
    v.attributes(disp[:,3], disp_color)
    v.set(point_size=point_size,
          show_axis=show_axis,
          bg_color=bg_color,
          show_grid=show_grid)

    return v


if __name__=='__main__':
    import pptk
    c1 = pptk.rand(100, 4)
    c2 = pptk.rand(100, 4)
    c3 = pptk.rand(100, 4)
    pptk_view3((c1, c2, c3))
    
