"""Decoding of heatmaps into landmark coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from charm.constants import LANDMARKS_PER_TOOTH


@dataclass
class DecodedLandmarks:
    """Landmarks decoded from a batch of heatmaps.

    ``coords`` (B, L, 3) are in millimetres; landmarks predicted absent sit
    on the placeholder. ``on_mesh`` (B, L) marks landmarks predicted on the
    scan, ``teeth`` (B, T) the predicted tooth presence and ``scores``
    (B, L) the heatmap value of the selected point.
    """

    coords: torch.Tensor
    on_mesh: torch.Tensor
    teeth: torch.Tensor
    scores: torch.Tensor


def to_world(
    points: torch.Tensor, centroid: torch.Tensor, tmaxv: torch.Tensor
) -> torch.Tensor:
    """Map normalized points (B, N, 3) back to millimetres."""
    return points * tmaxv.view(-1, 1, 1) + centroid.unsqueeze(1)


def decode_landmarks(
    heatmaps: torch.Tensor,
    points: torch.Tensor,
    centroid: torch.Tensor,
    tmaxv: torch.Tensor,
    presence: torch.Tensor | None = None,
    presence_gate: bool = True,
    threshold: float = 0.5,
) -> DecodedLandmarks:
    """Select the highest-scoring point of every heatmap.

    The placeholder is the last of the N points. With ``presence`` and
    ``presence_gate``, the five landmarks of a tooth are placed on the scan
    (at the best mesh point) when its presence probability reaches
    ``threshold`` and on the placeholder otherwise. Without gating, each
    landmark is absent when its heatmap peaks on the placeholder, and a
    tooth is present when any of its landmarks is on the scan.
    """
    B, L, N = heatmaps.shape
    placeholder = N - 1
    world = to_world(points, centroid, tmaxv)

    if presence is not None and presence_gate:
        teeth = presence >= threshold
        on_mesh = teeth.repeat_interleave(LANDMARKS_PER_TOOTH, dim=1)
        scores, idx = heatmaps[..., :placeholder].max(dim=-1)
    else:
        scores, idx = heatmaps.max(dim=-1)
        on_mesh = idx != placeholder
        teeth = on_mesh.view(B, -1, LANDMARKS_PER_TOOTH).any(dim=-1)

    idx = torch.where(on_mesh, idx, torch.full_like(idx, placeholder))
    coords = torch.gather(world, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    return DecodedLandmarks(coords, on_mesh, teeth, scores)
