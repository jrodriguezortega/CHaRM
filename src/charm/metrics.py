"""Presence-aware landmark metrics.

A predicted landmark is a true positive when it is predicted on the scan,
its tooth is present in the ground truth and it lies within ``threshold``
millimetres of the annotation. Precision divides the true positives by the
landmarks predicted on the scan and recall by the landmarks present in the
ground truth. Counts are pooled per dentition type; macro scores average
precision and recall over dentition types and take their harmonic mean,
micro scores pool every landmark. MEDE is the mean Euclidean distance over
landmarks present in both prediction and ground truth. Tooth presence is
scored with F1 = 2TP / (2TP + FP + FN) over teeth.
"""

from __future__ import annotations

import math

import torch
from torchmetrics import Metric

from charm.constants import DENTITION_TYPES, LANDMARKS_PER_TOOTH

COUNTS = (
    "tp",
    "n_pred",
    "n_gt",
    "dist_sum",
    "n_both",
    "teeth_tp",
    "teeth_fp",
    "teeth_fn",
)


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else math.nan


def _harmonic(a: float, b: float) -> float:
    if math.isnan(a) or math.isnan(b):
        return math.nan
    return 2 * a * b / (a + b) if a + b > 0 else 0.0


def threshold_tag(threshold: float) -> str:
    """Name suffix of a distance threshold, e.g. ``1mm``."""
    return f"{threshold:g}mm".replace(".", "")


def landmark_counts(
    pred_coords: torch.Tensor,
    pred_on_mesh: torch.Tensor,
    pred_teeth: torch.Tensor,
    gt_coords: torch.Tensor,
    gt_teeth: torch.Tensor,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Per-scan counts (B, len(COUNTS)) used by every metric.

    Coordinates are (B, L, 3) in millimetres, ``pred_on_mesh`` is (B, L)
    and the tooth presence tensors are (B, T).
    """
    gt_teeth = gt_teeth.bool()
    pred_teeth = pred_teeth.bool()
    gt_present = gt_teeth.repeat_interleave(LANDMARKS_PER_TOOTH, dim=1)
    both = pred_on_mesh & gt_present
    dist = torch.linalg.vector_norm(
        pred_coords.double() - gt_coords.double(), dim=-1
    )
    dist = torch.where(both, dist, torch.zeros_like(dist))
    counts = (
        (both & (dist <= threshold)).sum(1),
        pred_on_mesh.sum(1),
        gt_present.sum(1),
        dist.sum(1),
        both.sum(1),
        (pred_teeth & gt_teeth).sum(1),
        (pred_teeth & ~gt_teeth).sum(1),
        (~pred_teeth & gt_teeth).sum(1),
    )
    return torch.stack([c.double() for c in counts], dim=1)


def scores_from_counts(counts, tag: str = "1mm") -> dict[str, float]:
    """Precision, recall, F1, MEDE and presence F1 of pooled counts."""
    c = dict(zip(COUNTS, [float(v) for v in counts]))
    precision = _ratio(c["tp"], c["n_pred"])
    recall = _ratio(c["tp"], c["n_gt"])
    teeth = 2 * c["teeth_tp"] + c["teeth_fp"] + c["teeth_fn"]
    return {
        f"precision_{tag}": precision,
        f"recall_{tag}": recall,
        f"f1_{tag}": _harmonic(precision, recall),
        "mede": _ratio(c["dist_sum"], c["n_both"]),
        "f1_presence": _ratio(2 * c["teeth_tp"], teeth),
    }


class LandmarkMetrics(Metric):
    """Accumulate counts per dentition type and compute the scores.

    ``compute`` returns ``<metric>_macro``, ``<metric>_micro`` and the
    per-type ``<metric>/<dentition type>`` scores.
    """

    full_state_update = False
    higher_is_better = True

    def __init__(self, threshold: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.tag = threshold_tag(threshold)
        self.add_state(
            "counts",
            default=torch.zeros(
                len(DENTITION_TYPES), len(COUNTS), dtype=torch.float64
            ),
            dist_reduce_fx="sum",
        )

    def update(
        self,
        pred_coords: torch.Tensor,
        pred_on_mesh: torch.Tensor,
        pred_teeth: torch.Tensor,
        gt_coords: torch.Tensor,
        gt_teeth: torch.Tensor,
        dentition: torch.Tensor,
    ) -> None:
        counts = landmark_counts(
            pred_coords,
            pred_on_mesh,
            pred_teeth,
            gt_coords,
            gt_teeth,
            self.threshold,
        )
        self.counts.index_add_(0, dentition.long(), counts)

    def compute(self) -> dict[str, float]:
        counts = self.counts.cpu()
        results = {}
        per_type = []
        for t, row in enumerate(counts):
            if row[COUNTS.index("n_gt")] + row[COUNTS.index("n_pred")] == 0:
                continue
            scores = scores_from_counts(row, self.tag)
            per_type.append(scores)
            for name, value in scores.items():
                results[f"{name}/{DENTITION_TYPES[t]}"] = value

        def mean(name: str) -> float:
            values = [s[name] for s in per_type if not math.isnan(s[name])]
            return sum(values) / len(values) if values else math.nan

        for name in per_type[0] if per_type else ():
            results[f"{name}_macro"] = mean(name)
        if per_type:
            results[f"f1_{self.tag}_macro"] = _harmonic(
                results[f"precision_{self.tag}_macro"],
                results[f"recall_{self.tag}_macro"],
            )
        for name, value in scores_from_counts(counts.sum(0), self.tag).items():
            results[f"{name}_micro"] = value
        return results
