import os
from omegaconf import OmegaConf
import wandb


def setup_wandb(cfg):
    """
    Calls wandb.init() with the provided config.
    """
    # SLURM_JOB_ID is used to group runs in the same job together
    slurm_job_id = os.getenv("SLURM_JOB_ID", "local")

    # Initialize wandb
    wandb.init(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        group=slurm_job_id,  # Group by SLURM job ID
        job_type=cfg.logger.job_type,
        dir=f"{cfg.data.runs_dir}/{slurm_job_id}",  # Customize the directory if needed
        # offline=cfg.logger.offline,
        mode=cfg.logger.mode,
        # Log hyperparameters
        config=OmegaConf.to_container(cfg, resolve=True),
    )
