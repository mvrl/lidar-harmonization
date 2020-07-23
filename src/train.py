import warnings
warnings.filterwarnings('ignore')
from src.dataset.tools.callbacks.create_kde import CreateKDE
from src.dataset.tools.callbacks.create_tile import CreateTile
from src.harmonization.model import IntensityNet
from src.ex_pl.extended_trainer import ExtendedTrainer as Trainer
import code

if __name__ == '__main__':
    net = IntensityNet(
            "dataset/150_190000/train_dataset.csv",
            "dataset/150_190000/val_dataset.csv",
            "dataset/150_190000/test_dataset.csv",
            "dataset/big_tile/big_tile_dataset.csv",
            neighborhood_size=17,
            dual_flight=37).double()

    callbacks = [CreateKDE(), CreateTile()]
    trainer = Trainer(gpus=1, max_epochs=1, callbacks=callbacks)

    trainer.fit(net)
    trainer.test(net)
    trainer.qualitative_test(net)
    


