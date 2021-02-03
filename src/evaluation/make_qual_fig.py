import code
from pathlib import Path
import numpy as np
from pptk import viewer

def make_qual_fig(scan_num, inpath="fix_dublin_out"):
    inpath = Path(inpath)

    # load the fixed scans
    hm = inpath / "hm" / "shift" / (str(scan_num)+".txt.gz")
    dl = inpath / "dl" / "shift" / (str(scan_num)+".txt.gz")
    ref = inpath / "dl" / "shift" / "ref.txt.gz"

    if not (hm.exists() and dl.exists()):
        exit("can't find scans -- make sure to run fix_dublin_hm and fix_dublin_dl")

    hm_pc = np.loadtxt(hm)
    dl_pc = np.loadtxt(dl)
    ref = np.loadtxt(ref)

    v1 = viewer(dl_pc[:, :3])
    v1.attributes(dl_pc[:, 3], dl_pc[:, 4],  dl_pc[:, 5])

    v2 = viewer(hm_pc[:, :3])
    v2.attributes(hm_pc[:, 3], hm_pc[:, 5])
    for v in [v1, v2]:
        v.color_map("jet", scale=[0, 512])
        v.set(
            bg_color=[1, 1, 1, 1],
            show_grid=False,
            show_axis=False,
            show_info=False,
            lookat = [ 3.15249562e+05,  2.33671906e+05, -4.51619987e+01],
            r=1596.73828125,
            phi=-2.21506834,
            theta=1.52170897
        )  # set view

    v3 = viewer(ref[:, :3])
    v3.attributes(ref[:, 3])
    v3.color_map("jet", scale=[0, 512])
    v3.set(bg_color=[1,1,1,1], show_grid=False, show_axis=False,show_info=False)
    
    code.interact(local=locals())

    # fig, ax = plt.subplots(1, 3)
    
    # ax.flat[0].scatter(dl_pc[:, 0], dl_pc[:, 1], c=dl_pc[:, 3], s=1, vmin=0, vmax=512)
    # ax.flat[0].axis("off")

    # ax.flat[1].scatter(dl_pc[:, 0], dl_pc[:, 1], c=dl_pc[:, 5], s=1, vmin=0, vmax=512)
    # ax.flat[1].axis("off")

    # ax.flat[2].scatter(hm_pc[:, 0], hm_pc[:, 1], c=hm_pc[:, 5], s=1, vmin=0, vmax=512)
    # ax.flat[2].axis("off")

    # plt.show()

if __name__ == "__main__":
    # Sources: 30, 15, 6, 27, 21, 39, 26, 10, 0, 20, 40
    make_qual_fig(26)



    

