import os
from omegaconf import OmegaConf
import wandb
from socket import gethostname


def setup_wandb(cfg):
    """
    Calls wandb.init() with the provided config.
    """
    slurm_job_id = os.getenv("SLURM_JOB_ID") or os.getenv("SLURM_ARRAY_JOB_ID")
    wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group=slurm_job_id,
        name=f"{slurm_job_id}-{gethostname()}-{os.environ['SLURM_PROCID']}",
        job_type=cfg.logger.job_type,
        dir=f"{cfg.paths.runs_dir}/{slurm_job_id}",
        mode=cfg.logger.mode,
        # Log hyperparameters
        config=OmegaConf.to_container(cfg, resolve=True),
    )
