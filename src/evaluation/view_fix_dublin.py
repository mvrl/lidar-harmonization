import code
import numpy as np
from pathlib import Path
from pptk import viewer

def view_fix(path):
    # view the fix in path, where path contains two folders, default and shift
    #   path/default/{files.txt.gz}, path/shift/{files.txt.gz} where the files
    #   in both paths are the same. 

    path = Path(path)

    # validate paths here
    
    methods = {m.stem:m for m in path.glob("*")}
    print(methods)
    
    for m, p in methods.items():
        for type in ["default", "shift"]:
            print(f"visualizing {type}")
            scans_path = p / type
            dublin = np.empty((0, 7))
            for s in scans_path.glob("*.txt.gz"):
                scan = np.loadtxt(str(s))
                dublin = np.concatenate((dublin, scan), axis=0)

            v = viewer(dublin[:, :3])
            v.attributes(dublin[:, 3], dublin[:, 4], dublin[:, 5])
            v.color_map("jet", scale=[0, 512])
            MAE = np.mean(np.abs(dublin[:, 5] - dublin[:, 3]))/512
            print(f"{m} | {type}:", MAE)
            code.interact(local=locals())
    

if __name__ == "__main__":
    view_fix("fix_dublin_out")
    
