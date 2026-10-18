from pathlib import Path

import hydra
import pytest
from hydra import compose, initialize_config_dir

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
EXPERIMENTS = sorted(p.stem for p in (CONFIGS / "experiment").glob("*.yaml"))


def test_experiment_list():
    backbones = ("dgcnn", "pointmlp", "pointnet2", "pointnext")
    expected = {
        f"{b}_{v}"
        for b in (*backbones, "pointtransformer")
        for v in ("base", "charm")
    }
    expected |= {"pointmlp_postproc", "charnet"}
    assert set(EXPERIMENTS) == expected


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_compose_experiment(experiment):
    with initialize_config_dir(str(CONFIGS), version_base="1.3"):
        cfg = compose(
            "train",
            overrides=[f"experiment={experiment}"],
            return_hydra_config=True,
        )
    variant = cfg.model.variant
    if experiment != "charnet":
        assert experiment.endswith(variant)
    assert cfg.trainer.max_epochs == 200
    assert cfg.trainer.accumulate_grad_batches == 2
    assert sum(cfg.data.samples_per_dentition_type) == 10
    assert cfg.optimizer._target_ == "torch.optim.AdamW"
    assert cfg.callbacks.model_checkpoint.monitor == "val/f1_1mm_macro"
    assert hydra.utils.get_class(cfg.model._target_) is not None


def test_compose_eval():
    with initialize_config_dir(str(CONFIGS), version_base="1.3"):
        cfg = compose("eval", overrides=["ckpt_path=model.ckpt"])
    assert cfg.split == "test"
