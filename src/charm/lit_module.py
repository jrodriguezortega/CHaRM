"""Lightning module that trains and evaluates every landmark network."""

from __future__ import annotations

from typing import Any

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from charm.losses import CHaRMLoss
from charm.metrics import LandmarkMetrics
from charm.postprocess import decode_landmarks


def _plain(config: Any) -> Any:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    return config


class CHaRMModule(L.LightningModule):
    """Train a landmark network and report presence-aware metrics.

    ``net``, ``optimizer`` and ``scheduler`` are Hydra configurations
    (dictionaries with ``_target_``), stored as hyperparameters so that
    ``CHaRMModule.load_from_checkpoint`` rebuilds the network without any
    other file.
    """

    def __init__(
        self,
        net: dict,
        optimizer: dict,
        scheduler: dict | None = None,
        weight_seg: float = 1.0,
        weight_cls: float = 1.0,
        presence_gate: bool = True,
        threshold_mm: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "net": _plain(net),
                "optimizer": _plain(optimizer),
                "scheduler": _plain(scheduler),
                "weight_seg": weight_seg,
                "weight_cls": weight_cls,
                "presence_gate": presence_gate,
                "threshold_mm": threshold_mm,
            }
        )
        self.net = hydra.utils.instantiate(self.hparams.net)
        self.loss_fn = CHaRMLoss(weight_seg, weight_cls)
        self.train_metrics = LandmarkMetrics(threshold_mm)
        self.val_metrics = LandmarkMetrics(threshold_mm)
        self.test_metrics = LandmarkMetrics(threshold_mm)

    def metrics(self, stage: str) -> LandmarkMetrics:
        return getattr(self, f"{stage}_metrics")

    def forward(self, points: torch.Tensor, normals: torch.Tensor):
        return self.net(points, normals)

    def decode(self, batch: dict, heatmaps, presence):
        return decode_landmarks(
            heatmaps,
            batch["points"],
            batch["centroid"],
            batch["tmaxv"],
            presence,
            presence_gate=self.hparams.presence_gate,
        )

    def _step(self, batch: dict, stage: str):
        heatmaps, presence = self(batch["points"], batch["normals"])
        losses = self.loss_fn(
            heatmaps, batch["heatmaps"], presence, batch["presence"]
        )
        batch_size = batch["points"].shape[0]
        for name, value in losses.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=name == "loss",
                batch_size=batch_size,
            )
        with torch.no_grad():
            decoded = self.decode(batch, heatmaps.detach(), presence)
            self.metrics(stage).update(
                decoded.coords,
                decoded.on_mesh,
                decoded.teeth,
                batch["landmarks"],
                batch["presence"],
                batch["dentition"],
            )
        return losses["loss"]

    def training_step(self, batch: dict, batch_idx: int):
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int):
        self._step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int):
        self._step(batch, "test")

    def _log_metrics(self, stage: str) -> None:
        metrics = self.metrics(stage)
        results = metrics.compute()
        metrics.reset()
        main = (f"f1_{metrics.tag}_macro", "mede_macro", "f1_presence_macro")
        for name, value in results.items():
            if stage == "train" and "/" in name:
                continue
            self.log(
                f"{stage}/{name}",
                float(value),
                prog_bar=stage != "train" and name in main,
            )

    def on_train_epoch_end(self) -> None:
        self._log_metrics("train")

    def on_validation_epoch_end(self) -> None:
        self._log_metrics("val")

    def on_test_epoch_end(self) -> None:
        self._log_metrics("test")

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer, params=self.parameters()
        )
        if self.hparams.scheduler is None:
            return optimizer
        scheduler = hydra.utils.instantiate(
            self.hparams.scheduler, optimizer=optimizer
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
