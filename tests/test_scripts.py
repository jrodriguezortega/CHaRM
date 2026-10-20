import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("preprocess.py", "train.py", "evaluate.py", "predict.py")


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_help(name):
    result = subprocess.run(
        [sys.executable, ROOT / "scripts" / name, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout
    if name in ("preprocess.py", "predict.py"):
        assert f"scripts/{name}" in result.stdout
