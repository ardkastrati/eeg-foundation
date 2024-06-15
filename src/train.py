import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import hydra
import psutil
from lightning import Callback, LightningDataModule, LightningModule
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.accelerators import find_usable_cuda_devices
from omegaconf import DictConfig

import lightning as L
import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.models.mae_rope_module import MAEModuleRoPE
from src.models.mae_rope_net import ModularMaskedAutoencoderViTRoPE
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    task_wrapper,
    ProfilerCallback,
    setup_wandb,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """

    # set seed for random number generators in pytorch, numpy and python.random
    L.seed_everything(42, workers=True)

    # == Instantiate Loggers ==

    log.info("Instantiating loggers...")
    # logger = hydra.utils.instantiate(
    #     cfg.logger,
    #     dir=f"{cfg.paths.runs_dir}/{os.getenv('SLURM_JOB_ID')}",
    #     # group=f"{os.getenv('SLURM_JOB_ID')}",
    # )
    setup_wandb(cfg)

    # == Instantiate Callbacks ==

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating profiling callbacks...")
    callbacks.append(
        ProfilerCallback(
            runs_dir=cfg.paths.runs_dir,
            log_dir=cfg.debug.profile_dir,
            record_shapes=cfg.debug.record_shapes,
            with_stack=cfg.debug.with_stack,
            profile_memory=cfg.debug.profile_memory,
            log_epoch_freq=cfg.debug.log_epoch_freq,
        )
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # == Instantiate DataModule ==

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # == Instantiate Model ==

    log.info(f"Instantiating model <{cfg.model._target_}>")
    if cfg.restore_from_checkpoint and cfg.restore_from_checkpoint_path:
        checkpoint_path = cfg.restore_from_checkpoint_path
        if os.path.exists(checkpoint_path):
            log.info(f"Restoring model from checkpoint: {checkpoint_path}")
            # checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            # model = MAEModuleRoPE.load_from_checkpoint(checkpoint_path)
            # net: torch.nn.Module = hydra.utils.instantiate(cfg.model.net)
            # model.net = net
            # state_dict = {
            #     key.replace("net.", ""): value
            #     for key, value in checkpoint["state_dict"].items()
            # }
            # model.net.load_state_dict(state_dict)
            # model.optimizer = model.configure_optimizers()
            # model.optimizer.load_state_dict(checkpoint["optimizer_states"][0])
            # model.scheduler.load_state_dict(checkpoint["scheduler_states"][0])
            model: LightningModule = hydra.utils.instantiate(cfg.model)
        else:
            log.error(f"Checkpoint path does not exist: {checkpoint_path}")
            raise FileNotFoundError(
                f"Checkpoint path does not exist: {checkpoint_path}"
            )
    else:
        log.info("Starting training from scratch or checkpoint path not provided.")
        model: LightningModule = hydra.utils.instantiate(cfg.model)

    # == Instantiate Trainer ==

    # Check if a checkpoint path is provided and exists
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    if cfg.restore_from_checkpoint and os.path.isfile(cfg.restore_from_checkpoint_path):
        log.info(f"Resuming training from checkpoint: {cfg.restore_from_checkpoint}")
        trainer_cfg = dict(cfg.trainer)
        ckpt_path = cfg.restore_from_checkpoint_path
    else:
        trainer_cfg = cfg.trainer

    trainer = hydra.utils.instantiate(
        trainer_cfg,
        num_nodes=int(os.getenv("SLURM_JOB_NUM_NODES")),
        devices=(len(os.getenv("CUDA_VISIBLE_DEVICES").split(","))),
        callbacks=callbacks,
        # logger=logger,  # Uncomment if logger is configured
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "trainer": trainer,
    }

    # == Run Training ==

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path if cfg.restore_from_checkpoint else None,
        )
        # trainer.fit(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    # == Run Testing ==

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    print("At train.py entry", file=sys.stderr)
    print("RAM memory % used:", psutil.virtual_memory()[2], file=sys.stderr)
    print("RAM Used (GB):", psutil.virtual_memory()[3] / 1_000_000_000, file=sys.stderr)
    print("Usable cuda devices: ", find_usable_cuda_devices(), file=sys.stderr)

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
