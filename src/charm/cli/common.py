"""Helpers shared by the command line tools."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from charm.lit_module import CHaRMModule

REPO_CONFIGS = Path(__file__).resolve().parents[3] / "configs"


def config_dir() -> str:
    """Hydra configuration directory.

    ``CHARM_CONFIG_DIR`` if set, else ``./configs`` when it exists, else
    the ``configs`` folder of the repository.
    """
    if "CHARM_CONFIG_DIR" in os.environ:
        path = Path(os.environ["CHARM_CONFIG_DIR"])
    elif (Path.cwd() / "configs" / "train.yaml").is_file():
        path = Path.cwd() / "configs"
    else:
        path = REPO_CONFIGS
    return str(path.resolve())


def resolve_device(device: str = "auto") -> torch.device:
    """``auto`` selects the first GPU when available, else the CPU."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def load_checkpoint(path: str | Path, device: torch.device) -> CHaRMModule:
    """Load a trained module in evaluation mode."""
    module = CHaRMModule.load_from_checkpoint(str(path), map_location=device)
    return module.to(device).eval()
