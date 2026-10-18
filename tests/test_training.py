import json

import lightning as L
import numpy as np
import pytest
from conftest import NUM_POINTS, SCAN_ID

from charm.cli.common import load_checkpoint
from charm.cli.predict import main as predict
from charm.data import IOSLandmarksDataModule
from charm.data.preprocess import build_sample, sample_path
from charm.lit_module import CHaRMModule

OPTIMIZER = {"_target_": "torch.optim.AdamW", "lr": 0.005}
SCHEDULER = {
    "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
    "T_max": 2,
}


@pytest.fixture
def datamodule(tmp_path, scan):
    mesh_path, json_path, _, _ = scan
    data_dir = tmp_path / "processed"
    splits_dir = tmp_path / "splits"
    data_dir.mkdir()
    splits_dir.mkdir()
    for mirror in (False, True):
        sample = build_sample(
            mesh_path, json_path, mirror=mirror, num_points=NUM_POINTS
        )
        np.savez(sample_path(data_dir, SCAN_ID, mirror), **sample)
    for split in ("train", "val", "test"):
        (splits_dir / f"{split}.txt").write_text(SCAN_ID + "\n")
    return IOSLandmarksDataModule(
        data_dir=str(data_dir),
        splits_dir=str(splits_dir),
        samples_per_dentition_type=[0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
        eval_batch_size=1,
        num_workers=0,
        pin_memory=False,
    )


@pytest.mark.parametrize("variant", ["base", "postproc", "charm"])
def test_fit_test_predict(tmp_path, scan, datamodule, variant):
    module = CHaRMModule(
        net={"_target_": "charm.models.PointNet2", "variant": variant},
        optimizer=OPTIMIZER,
        scheduler=SCHEDULER,
        weight_seg=0.001,
    )
    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(module, datamodule=datamodule)
    (results,) = trainer.test(module, datamodule=datamodule, verbose=False)
    assert "test/f1_1mm_macro" in results
    assert "test/f1_presence/01" in results

    ckpt = tmp_path / "model.ckpt"
    trainer.save_checkpoint(ckpt)
    loaded = load_checkpoint(ckpt, module.device)
    assert loaded.net.variant == variant

    mesh_path = scan[0]
    out = tmp_path / "predictions"
    args = [str(ckpt), str(mesh_path), "-o", str(out), "--device", "cpu"]
    assert predict([*args, "--num-points", str(NUM_POINTS)]) == 0
    record = json.loads((out / f"{SCAN_ID}.json").read_text())
    assert record["key"] == SCAN_ID
    assert (out / f"{SCAN_ID}.fcsv").is_file()
