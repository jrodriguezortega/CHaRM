"""``charm-preprocess``: build the training samples of IOSLandmarks-1k."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from charm.data.dataset import read_split
from charm.data.preprocess import (
    NUM_POINTS,
    SEED,
    build_sample,
    sample_path,
)

logger = logging.getLogger("charm-preprocess")

SPLITS = ("train", "val", "test")


def default_output_dir(num_points: int) -> Path:
    return Path("data/processed") / f"points_{num_points}"


def process_one(
    dataset_dir: Path,
    output_dir: Path,
    scan_id: str,
    mirror: bool,
    num_points: int,
    seed: int,
) -> Path:
    sample = build_sample(
        dataset_dir / "meshes" / f"{scan_id}.vtp",
        dataset_dir / "annotations" / "json" / f"{scan_id}.json",
        mirror=mirror,
        num_points=num_points,
        seed=seed,
    )
    destination = sample_path(output_dir, scan_id, mirror)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez(temporary, **sample)
    temporary.replace(destination)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="charm-preprocess",
        description=(
            "Subsample the IOSLandmarks-1k meshes and store them with their "
            "landmarks. Training scans also get an x-mirrored copy."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/IOSLandmarks-1k"),
        help="Root of IOSLandmarks-1k (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/processed/points_<points>).",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=NUM_POINTS,
        help="Points per sample, placeholder included (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not write mirrored copies of the training scans.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel processes (default: %(default)s).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild samples that already exist.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    dataset_dir = args.dataset_dir.expanduser()
    output_dir = args.output_dir or default_output_dir(args.num_points)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for split in SPLITS:
        split_file = dataset_dir / "splits" / f"{split}.txt"
        if not split_file.is_file():
            logger.error("Missing split file %s", split_file)
            return 1
        mirrors = (
            (False,)
            if split != "train" or args.no_mirror
            else (
                False,
                True,
            )
        )
        for scan_id in read_split(split_file):
            for mirror in mirrors:
                path = sample_path(output_dir, scan_id, mirror)
                if args.overwrite or not path.exists():
                    jobs.append((scan_id, mirror))

    logger.info("Writing %d samples to %s", len(jobs), output_dir)
    common = (args.num_points, args.seed)
    failures = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                process_one, dataset_dir, output_dir, scan_id, mirror, *common
            ): (scan_id, mirror)
            for scan_id, mirror in jobs
        }
        for done, future in enumerate(as_completed(futures), start=1):
            scan_id, mirror = futures[future]
            try:
                future.result()
            except Exception:
                failures += 1
                logger.exception("Failed on %s (mirror=%s)", scan_id, mirror)
            if done % 100 == 0 or done == len(jobs):
                logger.info("%d / %d", done, len(jobs))

    if failures:
        logger.error("%d samples failed", failures)
        return 1
    logger.info("Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
