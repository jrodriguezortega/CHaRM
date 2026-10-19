import json

import numpy as np
import pytest
import torch
from conftest import NUM_POINTS, SCAN_ID

from charm.constants import LANDMARKS_PER_TOOTH, NUM_LANDMARKS, landmark_names
from charm.data import BalancedBatchSampler, IOSLandmarksDataset
from charm.data.preprocess import (
    build_sample,
    load_landmarks,
    mirror_order,
    sample_path,
)
from charm.io import write_fcsv


def test_json_round_trip(scan):
    _, json_path, coords, on_mesh = scan
    landmarks = load_landmarks(json_path, "lower")
    assert np.isnan(landmarks[~on_mesh]).all()
    np.testing.assert_allclose(landmarks[on_mesh], coords[on_mesh])
    record = json.loads(json_path.read_text())
    assert record["key"] == SCAN_ID
    assert record["landmarks"][0]["label_id"] == landmark_names("lower")[10]


def test_fcsv(tmp_path, scan):
    _, _, coords, on_mesh = scan
    path = write_fcsv(tmp_path / "scan.fcsv", "lower", coords, on_mesh)
    lines = path.read_text().splitlines()
    assert lines[0] == "# Markups fiducial file version = 4.10"
    rows = [line.split(",") for line in lines[3:]]
    assert len(rows) == on_mesh.sum()
    assert rows[0][11] == landmark_names("lower")[10]
    np.testing.assert_allclose(
        [float(v) for v in rows[0][1:4]], coords[10], atol=1e-5
    )


def test_build_sample(scan):
    mesh_path, json_path, coords, on_mesh = scan
    sample = build_sample(mesh_path, json_path, num_points=NUM_POINTS)

    assert sample["teeth_points"].shape == (NUM_POINTS, 3)
    np.testing.assert_array_equal(sample["teeth_points"][-1], [0, 0.5, 0])
    presence = sample["class_labels"]
    assert presence.tolist() == [0, 0] + [1] * 14
    placeholder = sample["centroid"] + np.array([0, sample["tmaxv"] / 2, 0])
    landmarks = sample["original_landmarks"]
    np.testing.assert_allclose(landmarks[0], placeholder, rtol=1e-6)
    np.testing.assert_allclose(landmarks[on_mesh], coords[on_mesh], rtol=1e-6)
    assert sample["heatmaps"].shape == (NUM_POINTS, NUM_LANDMARKS)
    assert sample["heatmaps"].dtype == np.float32
    assert float(sample["sigma"]) == 2.0

    again = build_sample(mesh_path, json_path, num_points=NUM_POINTS)
    np.testing.assert_array_equal(
        sample["teeth_points"], again["teeth_points"]
    )


def test_mirrored_sample(scan):
    mesh_path, json_path, coords, on_mesh = scan
    mirrored = build_sample(
        mesh_path, json_path, mirror=True, num_points=NUM_POINTS
    )
    assert mirrored["class_labels"].tolist() == [1] * 14 + [0, 0]
    expected = coords[mirror_order()] * np.array([-1, 1, 1])
    present = on_mesh[mirror_order()]
    np.testing.assert_allclose(
        mirrored["original_landmarks"][present], expected[present], rtol=1e-5
    )


def test_dataset_heatmaps(tmp_path, scan):
    mesh_path, json_path, _, _ = scan
    sample = build_sample(mesh_path, json_path, num_points=NUM_POINTS)
    np.savez(sample_path(tmp_path, SCAN_ID, False), **sample)

    item = IOSLandmarksDataset(tmp_path, [SCAN_ID])[0]
    heatmaps = item["heatmaps"]
    assert heatmaps.shape == (NUM_POINTS, NUM_LANDMARKS)
    absent = slice(0, 2 * LANDMARKS_PER_TOOTH)
    assert torch.all(heatmaps[:-1, absent] == 0)
    assert torch.all(heatmaps[-1, absent] == 1)
    assert torch.all(heatmaps[-1, 2 * LANDMARKS_PER_TOOTH :] == 0)
    assert heatmaps[:-1, 2 * LANDMARKS_PER_TOOTH :].max() > 0.5
    assert item["dentition"] == 1


def test_balanced_batch_sampler():
    groups = [list(range(0, 6)), list(range(6, 8)), []]
    sampler = BalancedBatchSampler(groups, [2, 1, 0])
    batches = list(sampler)
    assert len(batches) == len(sampler) == 3
    for batch in batches:
        assert len(batch) == 3
        assert sum(i < 6 for i in batch) == 2
    first = [i for batch in batches for i in batch if i < 6]
    assert sorted(first) == list(range(6))

    with pytest.raises(ValueError):
        BalancedBatchSampler(groups, [1, 1, 1])
