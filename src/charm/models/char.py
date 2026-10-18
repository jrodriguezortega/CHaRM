"""Conditioned Heatmap Regression (CHaR) shared by every backbone.

A backbone predicts one heatmap per landmark over the N input points, the
last of which is the placeholder that represents absent teeth. Depending on
``variant`` the network is:

``base``
    Heatmap regression only. Absent teeth must be inferred from heatmaps
    peaking on the placeholder.
``postproc``
    Heatmap regression plus a tooth-presence head trained jointly. The
    presence is applied to the predicted landmarks after decoding.
``charm``
    The presence probabilities condition the heatmaps inside the network:
    mesh points of tooth ``t`` are scaled by ``p_t`` and the placeholder by
    ``1 - p_t`` (CHaR).
"""

from __future__ import annotations

import torch
from torch import nn

from charm.constants import NUM_LANDMARKS, NUM_TEETH

VARIANTS = ("base", "postproc", "charm")


def combine_heatmaps(
    heatmaps: torch.Tensor, presence: torch.Tensor
) -> torch.Tensor:
    """Condition ``heatmaps`` (B, L, N) on tooth ``presence`` (B, T)."""
    B, L, N = heatmaps.shape
    T = presence.shape[1]
    per_tooth = heatmaps.view(B, T, L // T, N)
    p = presence[:, :, None, None]
    on_mesh = per_tooth[..., :-1] * p
    placeholder = per_tooth[..., -1:] * (1 - p)
    return torch.cat((on_mesh, placeholder), dim=-1).view(B, L, N)


def presence_head(
    in_features: int,
    hidden: tuple[int, ...],
    num_teeth: int,
    dropout: float,
) -> nn.Sequential:
    """MLP that maps a global feature to one presence logit per tooth."""
    layers = []
    for width in hidden:
        layers += [
            nn.Linear(in_features, width, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        in_features = width
    layers.append(nn.Linear(in_features, num_teeth))
    return nn.Sequential(*layers)


class LandmarkNet(nn.Module):
    """Base class of the landmark networks.

    Subclasses take ``points`` and ``normals`` of shape (B, N, 3) and return
    ``(heatmaps, presence)``: heatmaps of shape (B, num_landmarks, N) in
    [0, 1] and presence probabilities of shape (B, num_teeth), or ``None``
    for the ``base`` variant.
    """

    def __init__(
        self,
        variant: str = "charm",
        num_landmarks: int = NUM_LANDMARKS,
        num_teeth: int = NUM_TEETH,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if num_landmarks % num_teeth:
            raise ValueError("num_landmarks must be a multiple of num_teeth")
        self.variant = variant
        self.num_landmarks = num_landmarks
        self.num_teeth = num_teeth

    @property
    def has_presence_head(self) -> bool:
        return self.variant != "base"

    def finish(
        self,
        heatmaps: torch.Tensor,
        presence_logits: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the variant to the raw outputs of the backbone."""
        if not self.has_presence_head:
            return heatmaps, None
        presence = torch.sigmoid(presence_logits)
        if self.variant == "charm":
            heatmaps = combine_heatmaps(heatmaps, presence)
        return heatmaps, presence
