import math

import pytest
import torch

from charm.constants import LANDMARKS_PER_TOOTH, NUM_LANDMARKS, NUM_TEETH
from charm.losses import CHaRMLoss
from charm.metrics import LandmarkMetrics, landmark_counts, scores_from_counts
from charm.postprocess import decode_landmarks


def make_scan(present_teeth):
    gt_teeth = torch.zeros(1, NUM_TEETH, dtype=torch.bool)
    gt_teeth[0, present_teeth] = True
    gt = torch.arange(NUM_LANDMARKS * 3, dtype=torch.float64)
    return gt.view(1, NUM_LANDMARKS, 3), gt_teeth


def test_counts_and_scores():
    gt, gt_teeth = make_scan([0, 1])
    pred = gt.clone()
    pred[0, 0, 0] += 0.5
    pred[0, 1, 0] += 2.0
    pred_teeth = torch.zeros_like(gt_teeth)
    pred_teeth[0, [0, 2]] = True
    on_mesh = pred_teeth.repeat_interleave(LANDMARKS_PER_TOOTH, dim=1)

    counts = landmark_counts(pred, on_mesh, pred_teeth, gt, gt_teeth)
    scores = scores_from_counts(counts[0])

    # Tooth 0 is predicted with one landmark 2 mm off, tooth 2 is a false
    # positive and tooth 1 is missed.
    assert scores["precision_1mm"] == pytest.approx(4 / 10)
    assert scores["recall_1mm"] == pytest.approx(4 / 10)
    assert scores["f1_1mm"] == pytest.approx(0.4)
    assert scores["mede"] == pytest.approx(2.5 / 5)
    assert scores["f1_presence"] == pytest.approx(2 / 4)


def test_no_predictions_give_zero_precision():
    gt, gt_teeth = make_scan([0])
    pred_teeth = torch.zeros_like(gt_teeth)
    on_mesh = torch.zeros(1, NUM_LANDMARKS, dtype=torch.bool)
    counts = landmark_counts(gt, on_mesh, pred_teeth, gt, gt_teeth)
    scores = scores_from_counts(counts[0])
    assert scores["precision_1mm"] == 0.0
    assert scores["f1_1mm"] == 0.0
    assert math.isnan(scores["mede"])


def test_macro_averages_precision_and_recall_first():
    metrics = LandmarkMetrics()
    for dentition, present, predicted in ((0, [0, 1], [0]), (5, [0], [0])):
        gt, gt_teeth = make_scan(present)
        pred_teeth = torch.zeros_like(gt_teeth)
        pred_teeth[0, predicted] = True
        on_mesh = pred_teeth.repeat_interleave(LANDMARKS_PER_TOOTH, dim=1)
        metrics.update(
            gt, on_mesh, pred_teeth, gt, gt_teeth, torch.tensor([dentition])
        )
    results = metrics.compute()

    assert results["recall_1mm/00"] == pytest.approx(0.5)
    assert results["recall_1mm/10"] == pytest.approx(1.0)
    assert results["precision_1mm_macro"] == pytest.approx(1.0)
    assert results["recall_1mm_macro"] == pytest.approx(0.75)
    assert results["f1_1mm_macro"] == pytest.approx(2 * 0.75 / 1.75)
    assert results["recall_1mm_micro"] == pytest.approx(10 / 15)
    assert "f1_1mm/01" not in results


def test_decode_with_presence_gate():
    n = 6
    heatmaps = torch.zeros(1, NUM_LANDMARKS, n)
    heatmaps[..., -1] = 1.0
    heatmaps[..., 2] = 0.5
    points = torch.rand(1, n, 3)
    centroid = torch.tensor([[10.0, 20.0, 30.0]])
    tmaxv = torch.tensor([50.0])
    presence = torch.zeros(1, NUM_TEETH)
    presence[0, 3] = 0.9

    decoded = decode_landmarks(heatmaps, points, centroid, tmaxv, presence)
    world = points * 50.0 + centroid

    expected_teeth = presence >= 0.5
    assert torch.equal(decoded.teeth, expected_teeth)
    tooth = slice(3 * LANDMARKS_PER_TOOTH, 4 * LANDMARKS_PER_TOOTH)
    assert decoded.on_mesh[0, tooth].all()
    assert decoded.on_mesh.sum() == LANDMARKS_PER_TOOTH
    torch.testing.assert_close(
        decoded.coords[0, tooth], world[0, 2].expand(LANDMARKS_PER_TOOTH, 3)
    )
    torch.testing.assert_close(
        decoded.coords[0, 0], world[0, -1], msg="absent at the placeholder"
    )


def test_decode_without_gate():
    n = 6
    heatmaps = torch.zeros(1, NUM_LANDMARKS, n)
    heatmaps[..., -1] = 1.0
    heatmaps[0, 7, 1] = 2.0
    points = torch.rand(1, n, 3)
    decoded = decode_landmarks(
        heatmaps, points, torch.zeros(1, 3), torch.ones(1)
    )
    assert decoded.on_mesh.sum() == 1 and decoded.on_mesh[0, 7]
    assert decoded.teeth.sum() == 1 and decoded.teeth[0, 1]


def test_loss_weights():
    heatmaps = torch.rand(2, NUM_LANDMARKS, 5)
    target = torch.rand(2, 5, NUM_LANDMARKS)
    presence = torch.rand(2, NUM_TEETH)
    target_presence = (torch.rand(2, NUM_TEETH) > 0.5).float()

    losses = CHaRMLoss(0.001, 1.0)(heatmaps, target, presence, target_presence)
    seg = ((heatmaps - target.transpose(1, 2)) ** 2).sum() / 2
    cls = torch.nn.functional.binary_cross_entropy(presence, target_presence)
    torch.testing.assert_close(losses["loss_seg"], seg)
    torch.testing.assert_close(losses["loss"], 0.001 * seg + cls)
    assert "loss_cls" not in CHaRMLoss()(heatmaps, target)
