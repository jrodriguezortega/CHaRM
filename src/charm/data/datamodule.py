"""Lightning data module for IOSLandmarks-1k."""

from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from charm.constants import DENTITION_TYPES
from charm.data.dataset import IOSLandmarksDataset, read_split
from charm.data.sampler import BalancedBatchSampler


class IOSLandmarksDataModule(L.LightningDataModule):
    """Train, validation and test loaders of the published splits.

    Training batches hold ``samples_per_dentition_type[t]`` samples of each
    dentition type ``t`` (ordered as ``DENTITION_TYPES``), so the batch size
    is their sum. Heatmap targets have a standard deviation of ``sigma``
    millimetres. Validation and test use every scan once, without mirrored
    copies.
    """

    def __init__(
        self,
        data_dir: str = "data/processed/points_10001",
        splits_dir: str = "data/IOSLandmarks-1k/splits",
        sigma: float = 2.0,
        mirror_augmentation: bool = True,
        samples_per_dentition_type: list[int] | None = None,
        eval_batch_size: int = 10,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        super().__init__()
        if samples_per_dentition_type is None:
            samples_per_dentition_type = [1] * len(DENTITION_TYPES)
        if len(samples_per_dentition_type) != len(DENTITION_TYPES):
            raise ValueError(
                "samples_per_dentition_type needs one value per dentition "
                f"type {DENTITION_TYPES}"
            )
        self.save_hyperparameters(logger=False)
        self.datasets: dict[str, IOSLandmarksDataset] = {}

    @property
    def batch_size(self) -> int:
        return sum(self.hparams.samples_per_dentition_type)

    def _dataset(self, split: str) -> IOSLandmarksDataset:
        if split not in self.datasets:
            scan_ids = read_split(
                Path(self.hparams.splits_dir) / f"{split}.txt"
            )
            self.datasets[split] = IOSLandmarksDataset(
                self.hparams.data_dir,
                scan_ids,
                mirror_augmentation=(
                    split == "train" and self.hparams.mirror_augmentation
                ),
                sigma=self.hparams.sigma,
            )
        return self.datasets[split]

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self._dataset("train")
            self._dataset("val")
        if stage in (None, "validate"):
            self._dataset("val")
        if stage in (None, "test"):
            self._dataset("test")

    def train_dataloader(self) -> DataLoader:
        dataset = self._dataset("train")
        groups = [[] for _ in DENTITION_TYPES]
        for index in range(len(dataset)):
            groups[dataset.dentition_index(index)].append(index)
        sampler = BalancedBatchSampler(
            groups, self.hparams.samples_per_dentition_type
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def eval_dataloader(self, split: str) -> DataLoader:
        """Loader over every scan of ``split`` (val or test), in order."""
        if split not in ("val", "test"):
            raise ValueError(f"Expected split val or test, got {split!r}")
        return DataLoader(
            self._dataset(split),
            batch_size=self.hparams.eval_batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return self.eval_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self.eval_dataloader("test")
