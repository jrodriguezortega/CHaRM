import numpy as np
import pytest
import vedo

from charm.constants import LANDMARKS_PER_TOOTH, NUM_LANDMARKS
from charm.io import write_json

SCAN_ID = "01_0123456789ab_lower"
NUM_POINTS = 501


@pytest.fixture
def scan(tmp_path):
    """Synthetic lower arch with every tooth but the first two present."""
    mesh = vedo.Sphere(r=20.0, res=40).pos(1.0, 2.0, 3.0)
    mesh_path = tmp_path / f"{SCAN_ID}.vtp"
    mesh.write(str(mesh_path))

    vertices = np.asarray(mesh.vertices)
    rng = np.random.default_rng(0)
    coords = vertices[rng.choice(len(vertices), NUM_LANDMARKS)]
    on_mesh = np.ones(NUM_LANDMARKS, dtype=bool)
    on_mesh[: 2 * LANDMARKS_PER_TOOTH] = False
    json_path = write_json(
        tmp_path / f"{SCAN_ID}.json", SCAN_ID, "lower", coords, on_mesh
    )
    return mesh_path, json_path, coords, on_mesh
