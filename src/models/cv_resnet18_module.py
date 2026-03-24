import torch
from torch import nn
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
import torchvision.models as models


def compute_nme(preds, targets):
    B = preds.shape[0]

    preds = preds.view(B, -1, 2)
    targets = targets.view(B, -1, 2)

    diff = torch.norm(preds - targets, dim=2)
    mean_error = diff.mean(dim=1)

    # vẫn normalize theo landmark bbox
    x_min = targets[:, :, 0].min(dim=1).values
    x_max = targets[:, :, 0].max(dim=1).values
    y_min = targets[:, :, 1].min(dim=1).values
    y_max = targets[:, :, 1].max(dim=1).values

    norm = torch.sqrt((x_max - x_min) * (y_max - y_min)) + 1e-8

    nme = mean_error / norm

    return nme.mean()


class FaceLitModule(LightningModule):
    def __init__(
        self,
        net: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        num_landmarks: int = 98,
        compile: bool = False,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net

        # loss
        self.criterion = nn.MSELoss()

        # metrics
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_nme = MeanMetric()
        self.test_nme = MeanMetric()

        self.val_nme_best = MinMetric()

    def forward(self, x):
        return self.net(x)

    def model_step(self, batch):
        x, y = batch   
        preds = self(x)
        loss = self.criterion(preds, y)
        return loss, preds, y

    # ---------------- TRAIN ----------------
    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_epoch=True, prog_bar=True)

        return loss

    # ---------------- VALID ----------------
    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        nme = compute_nme(preds, targets)

        self.val_loss(loss)
        self.val_nme(nme)

        self.log("val/loss", self.val_loss, on_epoch=True, prog_bar=True)
        self.log("val/nme", self.val_nme, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        nme = self.val_nme.compute()
        self.val_nme_best(nme)

        self.log("val/nme_best", self.val_nme_best.compute(), prog_bar=True)

    # ---------------- TEST ----------------
    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)

        nme = compute_nme(preds, targets)

        self.test_loss(loss)
        self.test_nme(nme)

        self.log("test/loss", self.test_loss, on_epoch=True, prog_bar=True)
        self.log("test/nme", self.test_nme, on_epoch=True, prog_bar=True)

    # ---------------- SETUP ----------------
    def setup(self, stage: str):
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    # ---------------- OPTIM ----------------
    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/nme",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}