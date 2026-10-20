"""Detect the landmarks of new intraoral scans."""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from charm.cli.common import load_checkpoint, resolve_device
from charm.constants import ARCHES, arch_from_scan_id
from charm.data.preprocess import (
    NUM_POINTS,
    SEED,
    load_mesh,
    mesh_to_point_cloud,
    scan_seed,
)
from charm.io import write_fcsv, write_json
from charm.postprocess import decode_landmarks

log = logging.getLogger("predict")

MESH_SUFFIXES = (".vtp", ".stl", ".ply", ".obj")


def find_meshes(inputs: list[str]) -> list[Path]:
    """Mesh files given directly or found (non-recursively) in folders."""
    meshes = []
    for item in map(Path, inputs):
        if item.is_dir():
            meshes.extend(
                sorted(
                    p
                    for p in item.iterdir()
                    if p.suffix.lower() in MESH_SUFFIXES
                )
            )
        elif item.is_file() and item.suffix.lower() in MESH_SUFFIXES:
            meshes.append(item)
        else:
            raise FileNotFoundError(f"Not a mesh file or folder: {item}")
    if not meshes:
        raise FileNotFoundError("No .vtp, .stl, .ply or .obj meshes found")
    return meshes


def scan_arch(path: Path, arch: str | None) -> str:
    if arch is not None:
        return arch
    try:
        return arch_from_scan_id(path.stem)
    except ValueError:
        raise ValueError(
            f"Cannot tell the arch of {path.name}: name it *_upper/*_lower "
            "or pass --arch"
        ) from None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Predictor:
    """Preprocess a mesh and run the network on it."""

    def __init__(
        self,
        module,
        device: torch.device,
        num_points: int,
        seed: int,
        presence_gate: bool,
    ):
        self.module = module
        self.device = device
        self.num_points = num_points
        self.seed = seed
        self.presence_gate = presence_gate

    def preprocess(self, path: Path):
        rng = np.random.default_rng(scan_seed(self.seed, path.stem, False))
        cloud = mesh_to_point_cloud(load_mesh(path), self.num_points, rng)

        def tensor(array):
            return torch.as_tensor(
                np.asarray(array), dtype=torch.float32, device=self.device
            ).unsqueeze(0)

        return (
            tensor(cloud.points),
            tensor(cloud.normals),
            tensor(cloud.centroid),
            tensor(cloud.tmaxv),
        )

    @torch.no_grad()
    def infer(self, points, normals, centroid, tmaxv):
        heatmaps, presence = self.module(points, normals)
        return decode_landmarks(
            heatmaps,
            points,
            centroid,
            tmaxv.view(-1),
            presence,
            presence_gate=self.presence_gate,
        )


def benchmark(
    predictor: Predictor, meshes: list[Path], warmup: int, runs: int
) -> None:
    """Time preprocessing and inference (forward and decoding)."""
    inputs = predictor.preprocess(meshes[0])
    for _ in range(warmup):
        predictor.infer(*inputs)
    _sync(predictor.device)

    prep_times, infer_times = [], []
    for path in meshes:
        for _ in range(runs):
            start = time.perf_counter()
            inputs = predictor.preprocess(path)
            _sync(predictor.device)
            middle = time.perf_counter()
            predictor.infer(*inputs)
            _sync(predictor.device)
            end = time.perf_counter()
            prep_times.append(middle - start)
            infer_times.append(end - middle)

    def summary(name: str, times: list[float]) -> str:
        ms = [1000 * t for t in times]
        std = statistics.stdev(ms) if len(ms) > 1 else 0.0
        return f"{name}: {statistics.mean(ms):.1f} +/- {std:.1f} ms"

    total = [p + i for p, i in zip(prep_times, infer_times)]
    device = predictor.device
    name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    print(
        f"Benchmark on {name}, {len(meshes)} scan(s) x {runs} run(s), "
        f"{warmup} warm-up run(s)"
    )
    print("  " + summary("preprocessing", prep_times))
    print("  " + summary("inference", infer_times))
    print("  " + summary("total", total))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="scripts/predict.py",
        description=(
            "Detect the 80 landmarks of intraoral scans and write them as "
            "JSON (IOSLandmarks-1k format) and FCSV (3D Slicer). Landmarks "
            "of teeth predicted absent are not written."
        ),
    )
    parser.add_argument("checkpoint", type=Path, help="Trained .ckpt file")
    parser.add_argument(
        "inputs", nargs="+", help="Mesh files (.vtp/.stl/.ply/.obj) or folders"
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("predictions"),
        help="Output folder (default: %(default)s)",
    )
    parser.add_argument(
        "--arch",
        choices=ARCHES,
        default=None,
        help="Arch of every input (default: from the *_upper/*_lower name)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda or cuda:<index> (default: %(default)s)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=None,
        help="Points per scan, including the placeholder (default: the "
        f"training value, {NUM_POINTS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Seed of the vertex subsampling (default: %(default)s)",
    )
    parser.add_argument(
        "--no-presence-gate",
        action="store_true",
        help="Decode every landmark independently from its heatmap",
    )
    parser.add_argument(
        "--format",
        choices=("json", "fcsv", "both"),
        default="both",
        help="Output format (default: %(default)s)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Time preprocessing and inference instead of writing outputs",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warm-up runs of --benchmark (default: %(default)s)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Timed runs per scan of --benchmark (default: %(default)s)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)

    meshes = find_meshes(args.inputs)
    arches = {path: scan_arch(path, args.arch) for path in meshes}
    device = resolve_device(args.device)
    module = load_checkpoint(args.checkpoint, device)
    num_points = args.num_points or module.hparams.net.get(
        "points", NUM_POINTS
    )
    presence_gate = module.hparams.presence_gate and not args.no_presence_gate
    predictor = Predictor(module, device, num_points, args.seed, presence_gate)

    if args.benchmark:
        benchmark(predictor, meshes, args.warmup, args.runs)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in meshes:
        decoded = predictor.infer(*predictor.preprocess(path))
        coords = decoded.coords[0].cpu().numpy()
        on_mesh = decoded.on_mesh[0].cpu().numpy()
        arch = arches[path]
        if args.format in ("json", "both"):
            write_json(
                args.output_dir / f"{path.stem}.json",
                path.stem,
                arch,
                coords,
                on_mesh,
            )
        if args.format in ("fcsv", "both"):
            write_fcsv(
                args.output_dir / f"{path.stem}.fcsv", arch, coords, on_mesh
            )
        log.info(
            "%s: %d teeth, %d landmarks",
            path.name,
            int(decoded.teeth.sum()),
            int(decoded.on_mesh.sum()),
        )
    log.info("Predictions written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
