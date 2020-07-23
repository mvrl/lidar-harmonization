from src.dataset.tools.metrics import create_kde
from pytorch_lightning.callbacks import Callback

class CreateKDE(Callback):

    def __init__(self):
        pass

    # anywhere we want a KDE plot, we just put that here
    def on_train_end(self, trainer, pl_module):
        create_kde(
                pl_module.targets.flatten(),
                pl_module.predictions.flatten(),
                "Ground Truth",
                "Predictions (Training)",
                pl_module.results_dir / "train_kde_predictions.png",
                sample_size=5000,
                text=f"MAE = {pl_module.mae}")
 
    def on_validation_end(self, trainer, pl_module):
        create_kde(
                pl_module.targets.flatten(),
                pl_module.predictions.flatten(),
                "Ground Truth",
                "Predictions (Validation)",
                pl_module.results_dir / "valid_kde_predictions.png",
                sample_size=5000,
                text=f"MAE = {pl_module.mae}")

    def on_test_end(self, trainer, pl_module):
        create_kde(
                pl_module.targets.flatten(),
                pl_module.predictions.flatten(),
                "Ground Truth",
                "Predictions (Testing)",
                pl_module.results_dir / "test_kde_predictions.png",
                sample_size=5000,
                text=f"MAE = {pl_module.mae}")
    
    def on_qual_end(self, trainer, pl_module):
        create_kde(
                pl_module.targets.flatten(),
                pl_module.predictions.flatten(),
                "Ground Truth",
                "Predictions (Qualitative)",
                pl_module.results_dir / "qual_kde_predictions.png",
                sample_size=5000,
                text=f"MAE = {pl_module.mae}")


