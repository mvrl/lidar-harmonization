import warnings
warnings.filterwarnings('ignore')
from src.dataset.tools.callbacks.create_kde import CreateKDE
from src.dataset.tools.callbacks.create_tile import CreateTile
from src.harmonization.model import HarmonizationNet
from src.ex_pl.extended_trainer import ExtendedTrainer as Trainer
import code

if __name__ == '__main__':
    net = HarmonizationNet(
            "dataset/150_190000/train_dataset.csv",
            "dataset/150_190000/val_dataset.csv",
            "dataset/150_190000/test_dataset.csv",
            # The of the following is in-sample just like the test dataset before it
            # so it makes more sense to only do the no_overlap:
            #"dataset/big_tile_in_overlap/big_tile_dataset.csv",

            "dataset/big_tile_no_overlap/big_tile_dataset.csv",
            neighborhood_size=2,
            model_name="simple_mlp",
            dual_flight=None).double()  # scan 37 has the most examples 

    callbacks = [CreateKDE(), CreateTile()]
    trainer = Trainer(gpus=1, max_epochs=50, callbacks=callbacks)

    trainer.fit(net)
    trainer.test(net)
    trainer.qualitative_test(net)
    
