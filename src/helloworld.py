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
from lightning.pytorch.accelerators import find_usable_cuda_devices

import sys

if __name__ == "__main__":
    print(sys.path)
    print("hello world")
