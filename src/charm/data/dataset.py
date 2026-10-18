"""Dataset of preprocessed IOSLandmarks-1k samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from charm.constants import DENTITION_TYPES, dentition_type
from charm.data.preprocess import SIGMA, gaussian_heatmaps


def read_split(path: str | Path) -> list[str]:
    """Scan identifiers listed in a split file, one per line."""
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


class IOSLandmarksDataset(Dataset):
    """Samples written by ``charm-preprocess``.

    Each item is a dictionary with the normalized ``points`` and ``normals``
    (N x 3), the ``heatmaps`` target (N x 80) of standard deviation
    ``sigma`` millimetres, the tooth ``presence`` (16),
    the ground-truth ``landmarks`` in millimetres (80 x 3), the
    normalization parameters ``centroid`` and ``tmaxv``, the ``scan_id`` and
    the index of its ``dentition`` type.
    """

    def __init__(
        self,
        data_dir: str | Path,
        scan_ids: list[str],
        mirror_augmentation: bool = False,
        sigma: float = SIGMA,
    ):
        self.data_dir = Path(data_dir)
        self.sigma = sigma
        self.items = []
        for scan_id in scan_ids:
            self.items.append((scan_id, f"{scan_id}.npz"))
            if mirror_augmentation:
                self.items.append((scan_id, f"{scan_id}_mirror.npz"))
        missing = [
            name
            for _, name in self.items
            if not (self.data_dir / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} samples missing in {self.data_dir}, e.g. "
                f"{missing[0]}. Run charm-preprocess first."
            )

    def __len__(self) -> int:
        return len(self.items)

    def dentition_index(self, index: int) -> int:
        """Index in ``DENTITION_TYPES`` of the dentition of item ``index``."""
        return DENTITION_TYPES.index(dentition_type(self.items[index][0]))

    def __getitem__(self, index: int) -> dict:
        scan_id, name = self.items[index]
        with np.load(self.data_dir / name) as data:
            sample = {key: data[key] for key in data.files}
        tmaxv = float(sample["tmaxv"])
        heatmaps = gaussian_heatmaps(
            sample["teeth_points"].astype(np.float64) * tmaxv,
            sample["original_landmarks"].astype(np.float64)
            - sample["centroid"].astype(np.float64),
            sample["class_labels"],
            self.sigma,
        )
        return {
            "points": torch.from_numpy(sample["teeth_points"]),
            "normals": torch.from_numpy(sample["teeth_normals"]),
            "heatmaps": torch.from_numpy(heatmaps.astype(np.float32)),
            "presence": torch.from_numpy(
                sample["class_labels"].astype(np.float32)
            ),
            "landmarks": torch.from_numpy(sample["original_landmarks"]),
            "centroid": torch.from_numpy(sample["centroid"]),
            "tmaxv": torch.tensor(tmaxv),
            "scan_id": Path(name).stem,
            "dentition": self.dentition_index(index),
        }
