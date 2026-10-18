"""Writers for predicted landmarks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from charm.constants import LANDMARK_CLASSES, landmark_names

FCSV_HEADER = (
    "# Markups fiducial file version = 4.10\n"
    "# CoordinateSystem = RAS\n"
    "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,"
    "associatedNodeID\n"
)


def landmark_records(
    arch: str, coords: np.ndarray, on_mesh: Sequence[bool]
) -> list[dict]:
    """Landmarks predicted on the scan, as in ``labels_schema.json``."""
    records = []
    for uuid, label_id in enumerate(landmark_names(arch)):
        if not on_mesh[uuid]:
            continue
        records.append(
            {
                "coord": [float(v) for v in coords[uuid]],
                "uuid": uuid,
                "label_id": label_id,
                "class": LANDMARK_CLASSES[label_id.split("_")[1]],
            }
        )
    return records


def write_json(
    path: str | Path,
    scan_id: str,
    arch: str,
    coords: np.ndarray,
    on_mesh: Sequence[bool],
) -> Path:
    """Write landmarks in the IOSLandmarks-1k JSON annotation format."""
    path = Path(path)
    record = {
        "key": scan_id,
        "landmarks": landmark_records(arch, coords, on_mesh),
    }
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path


def write_fcsv(
    path: str | Path,
    arch: str,
    coords: np.ndarray,
    on_mesh: Sequence[bool],
) -> Path:
    """Write landmarks as a 3D Slicer markups fiducial file (v4.10)."""
    path = Path(path)
    lines = [FCSV_HEADER]
    for i, record in enumerate(landmark_records(arch, coords, on_mesh)):
        x, y, z = record["coord"]
        lines.append(
            f"vtkMRMLMarkupsFiducialNode_{i},{x:.6f},{y:.6f},{z:.6f},"
            f"0,0,0,1,1,1,1,{record['label_id']},,\n"
        )
    path.write_text("".join(lines))
    return path
