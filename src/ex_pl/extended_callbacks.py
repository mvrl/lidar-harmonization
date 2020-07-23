from pytorch_lightning.callbacks.base import Callback

# As it turns out, these actually need to be added to the source code directly. 
# Otherwise, we have to extend every callback, which is just not very feasible. 

class ExtendedCallback(Callback):

    def on_qual_batch_start(self, trainer, pl_module):
        """Call on qual batch start"""
        pass

    def on_qual_batch_end(self, trainer, pl_module):
        pass

    def on_qual_start(self, trainer, pl_module):
        pass

    def on_qual_end(self, trainer, pl_module):
        pass
