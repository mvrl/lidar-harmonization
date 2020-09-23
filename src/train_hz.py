import warnings
warnings.filterwarnings('ignore')
from src.dataset.tools.callbacks.create_kde import CreateKDE
from src.dataset.tools.callbacks.create_tile import CreateTile
from src.harmonization.model import HarmonizationNet
from src.interpolation.model import InterpolationNet
from src.ex_pl.extended_trainer import ExtendedTrainer as Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import code

if __name__ == '__main__':
    
    net = HarmonizationNet(
            "dataset/combined/train_dataset.csv",
            "dataset/combined/val_dataset.csv",
            "dataset/combined/test_dataset.csv",
            "dataset/big_tile_no_overlap/big_tile_dataset.csv",
            neighborhood_size=5,
            model_name="pointnet1",
            dual_flight=None).double()

    callbacks = [
            CreateKDE(), 
            CreateTile(), 
            ModelCheckpoint()]

    trainer = Trainer(
            gpus=1, 
            max_epochs=100,
            resume_from_checkpoint="lightning_logs/version_42/checkpoints/epoch=95.ckpt",
            callbacks=callbacks)

    # trainer.fit(net)
    # trainer.test(net)

    trainer.qualitative_test(net)
    
