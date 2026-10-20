"""Train a landmark network with the Hydra configs in ``configs/``."""

from __future__ import annotations

import logging

import hydra
import lightning as L
from omegaconf import DictConfig, OmegaConf

from charm.cli.common import config_dir
from charm.lit_module import CHaRMModule

log = logging.getLogger("train")


def train(cfg: DictConfig) -> dict[str, float]:
    L.seed_everything(cfg.seed, workers=True)

    datamodule = hydra.utils.instantiate(cfg.data)
    module = CHaRMModule(
        net=cfg.model,
        optimizer=cfg.optimizer,
        scheduler=cfg.get("scheduler"),
        weight_seg=cfg.loss.weight_seg,
        weight_cls=cfg.loss.weight_cls,
        presence_gate=cfg.presence_gate,
        threshold_mm=cfg.threshold_mm,
    )
    callbacks = [
        hydra.utils.instantiate(c)
        for c in (cfg.get("callbacks") or {}).values()
    ]
    loggers = [
        hydra.utils.instantiate(c) for c in (cfg.get("logger") or {}).values()
    ]
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers or False
    )
    for experiment_logger in loggers:
        experiment_logger.log_hyperparams(
            {
                "task_name": cfg.task_name,
                "seed": cfg.seed,
                "tags": list(cfg.tags),
                "data": OmegaConf.to_container(cfg.data, resolve=True),
                "trainer": OmegaConf.to_container(cfg.trainer, resolve=True),
            }
        )

    log.info("Training %s", cfg.task_name)
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    metrics = dict(trainer.callback_metrics)

    if cfg.test:
        best = getattr(trainer.checkpoint_callback, "best_model_path", "")
        if best:
            log.info("Testing best checkpoint %s", best)
        else:
            log.warning("No best checkpoint, testing the current weights")
        trainer.test(module, datamodule=datamodule, ckpt_path=best or None)
        metrics.update(trainer.callback_metrics)

    return {name: float(value) for name, value in metrics.items()}


def main() -> None:
    hydra.main(
        version_base="1.3", config_path=config_dir(), config_name="train"
    )(train)()


if __name__ == "__main__":
    main()
