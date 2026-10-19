"""Conversion of IOSLandmarks-1k scans into fixed-size samples.

Every mesh is subsampled to ``num_points - 1`` vertices, centred on the
centroid of the full mesh and scaled by the largest extent of the subsample
(``tmaxv``). A placeholder point is appended at ``(0, 0.5, 0)`` in normalized
coordinates and receives the landmarks of absent teeth. The training target
of every landmark is a Gaussian heatmap over the points, computed in
millimetres by ``gaussian_heatmaps`` and stored with the sample.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vedo

from charm.constants import (
    LANDMARKS_PER_TOOTH,
    NUM_LANDMARKS,
    arch_from_scan_id,
    landmark_names,
)

NUM_POINTS = 10001
SIGMA = 2.0
SEED = 42


@dataclass
class PointCloud:
    """Normalized point cloud of a scan, with the placeholder last."""

    points: np.ndarray
    normals: np.ndarray
    centroid: np.ndarray
    tmaxv: float
    indices: np.ndarray

    @property
    def placeholder(self) -> np.ndarray:
        """Placeholder position in the coordinate frame of the mesh."""
        return self.centroid + np.array([0.0, self.tmaxv / 2.0, 0.0])

    def to_world(self, points: np.ndarray) -> np.ndarray:
        """Map normalized points back to the coordinate frame of the mesh."""
        return points * self.tmaxv + self.centroid


def scan_seed(seed: int, scan_id: str, mirror: bool) -> np.random.SeedSequence:
    """Seed of one sample, independent of the processing order."""
    return np.random.SeedSequence(
        [seed, zlib.crc32(scan_id.encode()), int(mirror)]
    )


def mirror_order() -> np.ndarray:
    """Landmark order after mirroring: left and right teeth are swapped."""
    teeth = np.arange(NUM_LANDMARKS).reshape(-1, LANDMARKS_PER_TOOTH)
    return teeth[::-1].reshape(-1)


def sample_indices(
    n_vertices: int, n_points: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample ``n_points`` vertex indices, repeating on small meshes."""
    indices = rng.permutation(n_vertices)
    while indices.shape[0] < n_points:
        if indices.shape[0] * 2 < n_points:
            indices = np.concatenate((indices, indices))
        else:
            indices = np.concatenate(
                (indices, indices[: n_points - indices.shape[0]])
            )
    return indices[:n_points]


def load_mesh(path: str | Path, mirror: bool = False) -> vedo.Mesh:
    """Read a mesh and compute its vertex normals."""
    mesh = vedo.load(str(path))
    if mesh is None:
        raise ValueError(f"Could not read mesh {path}")
    mesh = mesh.compute_normals()
    if mirror:
        mesh.mirror(axis="x")
    return mesh


def mesh_to_point_cloud(
    mesh: vedo.Mesh, num_points: int, rng: np.random.Generator
) -> PointCloud:
    """Subsample, centre and normalize a mesh, appending the placeholder."""
    if num_points <= 1:
        raise ValueError("num_points must include the placeholder point")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    centroid = vertices.mean(axis=0)

    chosen = sample_indices(len(vertices), num_points - 1, rng)
    points = vertices[chosen] - centroid
    point_normals = normals[chosen]

    extent = points.max(axis=0) - points.min(axis=0)
    tmaxv = float(extent.max())
    points = np.vstack((points / tmaxv, [0.0, 0.5, 0.0]))
    point_normals = np.vstack((point_normals, np.ones(3)))
    if np.all(points[-1] == points[:-1], axis=1).any():
        raise RuntimeError("The placeholder coincides with a mesh point")
    return PointCloud(points, point_normals, centroid, tmaxv, chosen)


def load_landmarks(json_path: str | Path, arch: str) -> np.ndarray:
    """Landmarks of a JSON annotation in canonical order, NaN when absent.

    A tooth is present only when all five of its landmarks are annotated.
    """
    order = landmark_names(arch)
    with open(json_path) as f:
        annotation = json.load(f)

    by_label = {}
    for landmark in annotation["landmarks"]:
        label_id = landmark["label_id"]
        if label_id in by_label:
            raise ValueError(f"Duplicated landmark {label_id} in {json_path}")
        by_label[label_id] = landmark["coord"]

    unknown = sorted(set(by_label) - set(order))
    if unknown:
        raise ValueError(f"Unknown landmarks in {json_path}: {unknown}")

    coords = np.full((len(order), 3), np.nan, dtype=np.float64)
    for index, label_id in enumerate(order):
        if label_id in by_label:
            coords[index] = by_label[label_id]

    present = ~np.isnan(coords).any(axis=1)
    per_tooth = present.reshape(-1, LANDMARKS_PER_TOOTH)
    partial = per_tooth.any(axis=1) & ~per_tooth.all(axis=1)
    if partial.any():
        tooth = order[LANDMARKS_PER_TOOTH * int(np.argmax(partial))]
        raise ValueError(
            f"{json_path}: tooth {tooth.split('_')[0]} is partially annotated"
        )
    return coords


def gaussian_heatmaps(
    points_mm: np.ndarray,
    landmarks_mm: np.ndarray,
    presence: np.ndarray,
    sigma: float = SIGMA,
) -> np.ndarray:
    """Heatmap targets of shape ``(num_points, NUM_LANDMARKS)``.

    ``points_mm`` and ``landmarks_mm`` share a frame in millimetres, with
    the placeholder as the last point. For present teeth the placeholder
    target is 0; for absent teeth every mesh point is 0 and the placeholder
    is 1.
    """
    distances = np.linalg.norm(
        points_mm[:, None, :] - landmarks_mm[None, :, :], axis=2
    )
    heatmaps = np.exp(-0.5 * distances**2 / sigma**2)
    landmark_presence = np.repeat(presence.astype(bool), LANDMARKS_PER_TOOTH)
    heatmaps[-1, landmark_presence] = 0.0
    heatmaps[:-1, ~landmark_presence] = 0.0
    heatmaps[-1, ~landmark_presence] = 1.0
    return heatmaps


def build_sample(
    mesh_path: str | Path,
    json_path: str | Path,
    mirror: bool = False,
    num_points: int = NUM_POINTS,
    seed: int = SEED,
    sigma: float = SIGMA,
) -> dict[str, np.ndarray]:
    """Build the sample of one scan, optionally mirrored in x."""
    scan_id = Path(mesh_path).stem
    arch = arch_from_scan_id(scan_id)
    rng = np.random.default_rng(scan_seed(seed, scan_id, mirror))
    cloud = mesh_to_point_cloud(load_mesh(mesh_path, mirror), num_points, rng)

    landmarks = load_landmarks(json_path, arch)
    present = ~np.isnan(landmarks).any(axis=1)
    if mirror:
        landmarks[present, 0] *= -1
    landmarks[~present] = cloud.placeholder
    presence = present[::LANDMARKS_PER_TOOTH].astype(np.int64)
    if mirror:
        landmarks = landmarks[mirror_order()]
        presence = presence[::-1].copy()

    points = cloud.points.astype(np.float32)
    centroid = cloud.centroid.astype(np.float32)
    tmaxv = np.float32(cloud.tmaxv)
    landmarks = landmarks.astype(np.float32)
    heatmaps = gaussian_heatmaps(
        points.astype(np.float64) * float(tmaxv),
        landmarks.astype(np.float64) - centroid.astype(np.float64),
        presence,
        sigma,
    )
    return {
        "teeth_points": points,
        "teeth_normals": cloud.normals.astype(np.float32),
        "centroid": centroid,
        "tmaxv": tmaxv,
        "original_landmarks": landmarks,
        "class_labels": presence,
        "heatmaps": heatmaps.astype(np.float32),
        "sigma": np.float32(sigma),
    }


def sample_path(output_dir: str | Path, scan_id: str, mirror: bool) -> Path:
    """Location of the sample of ``scan_id`` inside ``output_dir``."""
    suffix = "_mirror" if mirror else ""
    return Path(output_dir) / f"{scan_id}{suffix}.npz"
