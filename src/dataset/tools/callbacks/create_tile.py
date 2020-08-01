from pytorch_lightning.callbacks import Callback
import code
import pptk
import numpy as np

class CreateTile(Callback):

    def __init__(self):
        pass

    # at the end of the qualitative evaluation step, create the big tile and
    # render it using pptk, then save it as an npy file

    def on_qual_batch_end(self, trainer, pl_module):
        # somehow there is a way to move the code from qual_step to this block
        # and preserve the readability in model.py
        pass

    def on_qual_end(self, trainer, pl_module):
        pl_module.xyzi = pl_module.xyzi.numpy()
        np.save(pl_module.results_dir / "tile.npy", pl_module.xyzi)

        # Load the other half if in-overlap
        if (pl_module.qual_dataset_csv.parents[0] / "base_flight_tile.npy").exists():
            print("found base tile for in-overlap dataset")
            base = np.load(pl_module.qual_dataset_csv.parents[0] / "base_flight_tile.npy") 

            # Concatenate these together:
            combined_tile = np.concatenate((base[:, :3], pl_module.xyzi[:, :3]))
        
            # Build attributes. Multiply the predictions by 512 to rescale to real values
            attr1 = np.concatenate((base[:, 3], pl_module.xyzi[:, 3]*512)) # alt
            attr2 = np.concatenate((base[:, 3], pl_module.xyzi[:, 4]*512)) # gt
            attr3 = np.concatenate((base[:, 3], pl_module.xyzi[:, 5]*512)) # prediction
            attr3 = np.clip(attr3, 0, 512) 

            v = pptk.viewer(combined_tile)

        # Otherwise just load the tile
        else:
            print("no base tile found!")
            attr1 = pl_module.xyzi[:, 3]*512 # alt
            attr2 = pl_module.xyzi[:, 4]*512 # gt
            attr3 = pl_module.xyzi[:, 5]*512 # prediction
            attr3 = np.clip(attr3, 0, 512)

            v = pptk.viewer(pl_module.xyzi[:, :3])

        v.attributes(attr1, attr2, attr3)
        v.set(bg_color=(1,1,1,1))
        v.set(show_grid=False)
        v.set(show_info=False)
        v.set(show_axis=False)
        v.set(theta = 1.53398085)
        v.set(phi = 1.61374784)
        v.set(r = 239.49221802)
        print("try to line up the image correctly!")
        qual_results_dir = pl_module.results_dir / "overlap" if pl_module.qual_in else pl_module.results_dir / "no_overlap"
        qual_results_dir.mkdir(parents=True, exist_ok=True)
        v.set(curr_attribute_id=0)
        
        v.capture(qual_results_dir / "qual_alt_capture.png")
        v.set(curr_attribute_id=1)
        v.capture(qual_results_dir / "qual_gt_capture.png")
        v.set(curr_attribute_id=2)
        v.capture(qual_results_dir / "qual_pred_capture.png")
        code.interact(local=locals())
        v.close()
