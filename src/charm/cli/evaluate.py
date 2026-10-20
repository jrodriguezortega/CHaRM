"""Score a checkpoint on the validation or test split."""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from charm.cli.common import config_dir, load_checkpoint, resolve_device
from charm.constants import DENTITION_TYPES, arch_from_scan_id
from charm.io import write_fcsv, write_json
from charm.metrics import (
    LandmarkMetrics,
    landmark_counts,
    scores_from_counts,
)
from charm.postprocess import decode_landmarks

log = logging.getLogger("evaluate")


def _clean(value: float) -> float | None:
    return None if math.isnan(value) else value


def per_type_table(results: dict[str, float], tag: str) -> list[dict]:
    """Rows of every dentition type plus the macro and micro averages."""
    names = (
        f"precision_{tag}",
        f"recall_{tag}",
        f"f1_{tag}",
        "mede",
        "f1_presence",
    )
    rows = []
    for dentition in DENTITION_TYPES:
        if f"f1_{tag}/{dentition}" in results:
            row = {"dentition_type": dentition}
            row.update({n: results[f"{n}/{dentition}"] for n in names})
            rows.append(row)
    for average in ("macro", "micro"):
        row = {"dentition_type": average}
        row.update({n: results.get(f"{n}_{average}") for n in names})
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(cfg: DictConfig) -> dict[str, float]:
    device = resolve_device(cfg.device)
    module = load_checkpoint(cfg.ckpt_path, device)
    presence_gate = cfg.presence_gate
    if presence_gate is None:
        presence_gate = module.hparams.presence_gate
    threshold = cfg.threshold_mm or module.hparams.threshold_mm

    datamodule = hydra.utils.instantiate(cfg.data)
    loader = datamodule.eval_dataloader(cfg.split)

    output_dir = Path(cfg.output_dir)
    prediction_dir = output_dir / "predictions"
    if cfg.save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    metrics = LandmarkMetrics(threshold).to(device)
    per_scan = []
    for batch in loader:
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        heatmaps, presence = module(batch["points"], batch["normals"])
        decoded = decode_landmarks(
            heatmaps,
            batch["points"],
            batch["centroid"],
            batch["tmaxv"],
            presence,
            presence_gate=presence_gate,
        )
        args = (
            decoded.coords,
            decoded.on_mesh,
            decoded.teeth,
            batch["landmarks"],
            batch["presence"],
        )
        metrics.update(*args, batch["dentition"])
        counts = landmark_counts(*args, threshold=threshold).cpu()

        coords = decoded.coords.cpu().numpy()
        on_mesh = decoded.on_mesh.cpu().numpy()
        for i, scan_id in enumerate(batch["scan_id"]):
            row = {
                "scan_id": scan_id,
                "dentition_type": scan_id[:2],
                "teeth_gt": int(batch["presence"][i].sum()),
                "teeth_pred": int(decoded.teeth[i].sum()),
            }
            row.update(scores_from_counts(counts[i], metrics.tag))
            per_scan.append(row)
            if cfg.save_predictions:
                arch = arch_from_scan_id(scan_id)
                write_json(
                    prediction_dir / f"{scan_id}.json",
                    scan_id,
                    arch,
                    coords[i],
                    on_mesh[i],
                )
                write_fcsv(
                    prediction_dir / f"{scan_id}.fcsv",
                    arch,
                    coords[i],
                    on_mesh[i],
                )

    results = metrics.compute()
    split = cfg.split
    with open(output_dir / f"metrics_{split}.json", "w") as f:
        json.dump(
            {
                "checkpoint": str(cfg.ckpt_path),
                "split": split,
                "presence_gate": bool(presence_gate),
                "threshold_mm": threshold,
                "metrics": {k: _clean(v) for k, v in results.items()},
            },
            f,
            indent=2,
        )
    table = per_type_table(results, metrics.tag)
    write_csv(output_dir / f"per_type_{split}.csv", table)
    write_csv(output_dir / f"per_scan_{split}.csv", per_scan)

    tag = metrics.tag
    log.info(
        "%s: F1@%s macro %.4f (P %.4f, R %.4f), micro %.4f, "
        "MEDE %.3f mm, presence F1 %.4f",
        split,
        tag,
        results[f"f1_{tag}_macro"],
        results[f"precision_{tag}_macro"],
        results[f"recall_{tag}_macro"],
        results[f"f1_{tag}_micro"],
        results["mede_macro"],
        results["f1_presence_macro"],
    )
    log.info("Results written to %s", output_dir)
    return results


def main() -> None:
    hydra.main(
        version_base="1.3", config_path=config_dir(), config_name="eval"
    )(evaluate)()


if __name__ == "__main__":
    main()
