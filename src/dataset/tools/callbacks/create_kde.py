from src.dataset.tools.metrics import create_kde, create_interpolation_harmonization_plot
from pytorch_lightning.callbacks import Callback

class CreateKDE(Callback):
    def __init__(self):
        pass

    # anywhere we want a KDE plot, we just put that here
    def on_train_epoch_end(self, trainer, pl_module):
        # why did this stop working?
        create_interpolation_harmonization_plot(
                pl_module.h_targets.flatten(),
                pl_module.h_preds.flatten(),
                pl_module.h_mae,
                pl_module.i_targets.flatten(),
                pl_module.i_preds.flatten(),
                pl_module.i_mae,
                "Train",
                pl_module.results_dir/ f"train_kde_{pl_module.neighborhood_size}.png")

    def on_train_end(self, trainer, pl_module):
        create_interpolation_harmonization_plot(
                pl_module.h_targets.flatten(),
                pl_module.h_preds.flatten(),
                pl_module.h_mae,
                pl_module.i_targets.flatten(),
                pl_module.i_preds.flatten(),
                pl_module.i_mae,
                "Train",
                pl_module.results_dir/ f"train_kde_{pl_module.neighborhood_size}.png")


    def on_validation_end(self, trainer, pl_module):
        create_interpolation_harmonization_plot(
                pl_module.h_targets.flatten(),
                pl_module.h_preds.flatten(),
                pl_module.h_mae,
                pl_module.i_targets.flatten(),
                pl_module.i_preds.flatten(),
                pl_module.i_mae,
                "Validation",
                pl_module.results_dir/ f"val_kde_{pl_module.neighborhood_size}.png")

    def on_test_end(self, trainer, pl_module):
        create_interpolation_harmonization_plot(
                pl_module.h_targets.flatten(),
                pl_module.h_preds.flatten(),
                pl_module.h_mae,
                pl_module.i_targets.flatten(),
                pl_module.i_preds.flatten(),
                pl_module.i_mae,
                "Testing",
                pl_module.results_dir/ f"test_kde_{pl_module.neighborhood_size}.png")

    def on_qual_end(self, trainer, pl_module):
        suffix = "_in_overlap.png" if pl_module.qual_in else "_no_overlap.png"
        create_kde(
                pl_module.h_targets.flatten(),
                pl_module.h_preds.flatten(),
                "Ground Truth - Harmonization",
                "Predictions (Qual)",
                pl_module.results_dir / (f"qual_kde_{pl_module.neighborhood_size}_i"+suffix),
                sample_size=5000,
                text=f"MAE = {pl_module.h_mae}")

        create_kde(
                pl_module.i_targets.flatten(),
                pl_module.i_preds.flatten(),
                "Ground Truth - Interpolation",
                "Predictions (Qual)",
                pl_module.results_dir / (f"qual_kde_{pl_module.neighborhood_size}_i"+suffix),
                sample_size=5000,
                text=f"MAE = {pl_module.i_mae}")

        create_interpolation_harmonization_plot(
                pl_module.h_targets.flatten(),
                pl_module.h_preds.flatten(),
                pl_module.h_mae,
                pl_module.i_targets.flatten(),
                pl_module.i_preds.flatten(),
                pl_module.i_mae,
                "Train",
                pl_module.results_dir/ f"qual_kde_{pl_module.neighborhood_size}.png")
