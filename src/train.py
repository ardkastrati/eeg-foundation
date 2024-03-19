import os
from typing import Any, Dict, List, Optional, Tuple
import psutil
import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
import multiprocessing
import tempfile
from lightning.pytorch.accelerators import find_usable_cuda_devices
from torch.autograd import profiler

from pytorch_lightning.profilers import (
    PyTorchProfiler,
    SimpleProfiler,
    AdvancedProfiler,
)

from pytorch_lightning.profilers.pytorch import ScheduleWrapper

from torch.profiler import tensorboard_trace_handler, ProfilerAction
import wandb

import time, socket
from datetime import datetime

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

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
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

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    # print(callbacks)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    # Setup the profiler with custom trace handler as a callback
    log.info(f"Instantiating debug (profiler)...")
    epoch_freq = 5
    log_dir = cfg.debug.profile_dir

    log.info("Instantiating profiling output directory...")
    now = datetime.now()
    dir_name = now.strftime("%Y-%m-%d_%H-%M")
    log_path = os.path.join(log_dir, dir_name)
    os.makedirs(log_path, exist_ok=True)

    def output_fn(profiler, worker_id, epoch):
        profiler.export_chrome_trace(
            f"{log_path}/{epoch}_{worker_id}_{socket.gethostname()}.pt.trace.json"
        )

    def profileTest(current_epoch: int, epoch_freq: int):
        if (
            current_epoch < 2
            or (5 <= current_epoch and current_epoch < 10)
            or current_epoch % epoch_freq == 0
        ):
            return True
        else:
            return False

    class ProfilerCallback(Callback):

        def on_train_epoch_start(self, trainer, pl_module):
            current_epoch = trainer.current_epoch
            nr_batches = len(trainer.train_dataloader)
            if profileTest(current_epoch=current_epoch, epoch_freq=epoch_freq):
                pl_module.profiler = torch.profiler.profile(
                    schedule=torch.profiler.schedule(
                        wait=1, warmup=1, active=nr_batches - 2, repeat=1
                    ),
                    on_trace_ready=lambda p: output_fn(
                        p, worker_id=trainer.global_rank, epoch=current_epoch
                    ),
                    record_shapes=cfg.debug.record_shapes,
                    with_stack=cfg.debug.with_stack,
                    profile_memory=cfg.debug.profile_memory,
                )
                print("Starting profiling")
                pl_module.profiler.start()
            pl_module.epoch_start_time = time.time()
            print("New epoch started")

        def on_train_epoch_end(self, trainer, pl_module, unused=None):
            print("Finishing up")
            current_epoch = trainer.current_epoch
            if profileTest(current_epoch=current_epoch, epoch_freq=epoch_freq):
                pl_module.profiler.stop()
                print("Finished profiling")
                del pl_module.profiler
            pl_module.epoch_end_time = time.time()
            epoch_train_time = pl_module.epoch_end_time - pl_module.epoch_start_time
            # wandb.log({"epoch_nr": current_epoch, "epoch_train_time": epoch_train_time})
            print(f"epoch_train_time: {epoch_train_time}")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            current_epoch = trainer.current_epoch
            if profileTest(current_epoch=current_epoch, epoch_freq=epoch_freq):
                pl_module.profiler.step()

    callbacks.append(ProfilerCallback())

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
        # "profiler": prof,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        start_time = time.time()
        # trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        trainer.fit(model=model, datamodule=datamodule)
        end_time = time.time()
        print(f"Finished training in {end_time - start_time}s!!!!")

    train_metrics = trainer.callback_metrics

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
    print("At train.py entry")
    print("RAM memory % used:", psutil.virtual_memory()[2])
    print("RAM Used (GB):", psutil.virtual_memory()[3] / 1000000000)
    print("Usable cuda devices: ", find_usable_cuda_devices())

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
    print("hello world")
    main()
