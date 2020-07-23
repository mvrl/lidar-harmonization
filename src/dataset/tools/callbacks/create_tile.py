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

        # Load the other half
        base = np.load("dataset/big_tile/base_flight_tile.npy") 

        # Concatenate these together:
        combined_tile = np.concatenate((base[:, :3], pl_module.xyzi[:, :3]))
        
        # Build attributes. Multiply the predictions by 512 to rescale to real values
        attr1 = np.concatenate((base[:, 3], pl_module.xyzi[:, 3]*512)) # alt
        attr2 = np.concatenate((base[:, 3], pl_module.xyzi[:, 4]*512)) # gt
        attr3 = np.concatenate((base[:, 3], pl_module.xyzi[:, 5]*512)) # prediction
        attr3 = np.clip(attr3, 0, 512) 

        v = pptk.viewer(combined_tile)
        v.attributes(attr1, attr2, attr3)
        code.interact(local=locals())

    

        # additionally we need the original xyz data where this data comes from


