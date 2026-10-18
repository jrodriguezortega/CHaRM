"""Training objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class CHaRMLoss(nn.Module):
    """Heatmap regression loss plus tooth-presence classification.

    The heatmap term is the squared error summed over landmarks and points
    and averaged over the batch. The presence term is the binary cross
    entropy of the presence probabilities, used only when the network
    predicts them.
    """

    def __init__(self, weight_seg: float = 1.0, weight_cls: float = 1.0):
        super().__init__()
        self.weight_seg = weight_seg
        self.weight_cls = weight_cls

    def forward(
        self,
        heatmaps: torch.Tensor,
        target_heatmaps: torch.Tensor,
        presence: torch.Tensor | None = None,
        target_presence: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """``heatmaps`` (B, L, N); ``target_heatmaps`` (B, N, L)."""
        batch_size = heatmaps.shape[0]
        target = target_heatmaps.transpose(1, 2)
        seg = F.mse_loss(heatmaps, target, reduction="sum") / batch_size
        loss = self.weight_seg * seg
        losses = {"loss_seg": seg}
        if presence is not None:
            cls = F.binary_cross_entropy(presence, target_presence)
            loss = loss + self.weight_cls * cls
            losses["loss_cls"] = cls
        losses["loss"] = loss
        return losses
