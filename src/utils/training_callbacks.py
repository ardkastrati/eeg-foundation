from datetime import datetime
import os
from socket import gethostname
from lightning import Callback
import torch
import wandb
import csv
from socket import gethostname
import time


class ProfilerCallback(Callback):
    """
    Callback for profiling the training process with a PyTorchProfiler.

    This callback enables profiling during training epochs and provides methods to export
    profiling results and log training metrics.

    Args:
        runs_dir (str): The directory where the training runs are stored.
        log_dir (str): The directory where the generated traces are stored.
        record_shapes (bool): Whether to record shapes during profiling.
        with_stack (bool): Whether to record stack traces during profiling.
        profile_memory (bool): Whether to profile memory usage.
        log_epoch_freq (int): The frequency at which to log profiling results.
        hostname (str): The hostname of the machine running the training.

    Attributes:
        run_dir (str): The directory for the current training run.
        log_dir (str): The directory for log files.
        record_shapes (bool): Whether to record shapes during profiling.
        with_stack (bool): Whether to record stack traces during profiling.
        profile_memory (bool): Whether to profile memory usage.
        epoch_freq (int): The frequency at which to log profiling results.
        log_path (str): The path for saving log files.
        hostname (str): The hostname of the machine running the training.
    """

    def __init__(
        self,
        runs_dir,
        log_dir,
        record_shapes,
        with_stack,
        profile_memory,
        log_epoch_freq,
    ):
        super().__init__()
        self.run_dir = f"{runs_dir}/{os.getenv('SLURM_JOB_ID')}"
        self.log_dir = log_dir
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.profile_memory = profile_memory
        self.epoch_freq = log_epoch_freq
        # self.log_path = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M"))
        self.log_path = os.path.join(log_dir, f"{os.getenv('SLURM_JOB_ID')}")
        os.makedirs(self.log_path, exist_ok=True)
        self.hostname = gethostname()

    def profileTest(self, current_epoch: int):
        """Check whether profiler should be active for this epoch.

        Args:
            current_epoch (int): The current epoch number.

        Returns:
            bool: True if the test should be profiled, False otherwise.
        """
        return False
        # if (
        #     # current_epoch < 20
        #     current_epoch < 2
        #     or (5 <= current_epoch and current_epoch < 10)
        #     or current_epoch % self.epoch_freq == 0
        # ):
        #     return True
        # else:
        #     return False

    def output_fn(self, profiler, worker_id, epoch):
        """
        Export the profiler's chrome trace to a file.

        Args:
            profiler (Profiler): The profiler object.
            worker_id (int): The global_rank of the worker.
            epoch (int): The current epoch.

        Returns:
            None
        """
        profiler.export_chrome_trace(
            f"{self.log_path}/{epoch}_{worker_id}_{self.hostname}.pt.trace.json"
        )

    def on_train_epoch_start(self, trainer, pl_module):
        """
        Callback function called at the start of each training epoch.

        - Starts the profiler if it is active.
        - Logs the epoch start time.

        Args:
            trainer: The PyTorch Lightning Trainer object.
            pl_module: The PyTorch Lightning module being trained.

        Returns:
            None
        """
        current_epoch = trainer.current_epoch
        nr_batches = len(trainer.train_dataloader)

        wandb.log({"epoch": current_epoch}, step=trainer.global_step)

        if self.profileTest(current_epoch=current_epoch):
            pl_module.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
                on_trace_ready=lambda p: self.output_fn(
                    p, worker_id=trainer.global_rank, epoch=current_epoch
                ),
                record_shapes=self.record_shapes,
                with_stack=self.with_stack,
                profile_memory=self.profile_memory,
            )
            print("Starting profiling")
            pl_module.profiler.start()

        pl_module.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, unused=None):
        """
        Callback function called at the end of each training epoch.

        - Stops the profiler if it is active.
        - Logs the epoch training time.
        - Logs the throughput of the epoch (in Spectograms/s).

        Args:
            trainer: The PyTorch Lightning trainer object.
            pl_module: The PyTorch Lightning module being trained.
            unused: Unused argument.

        Returns:
            None
        """
        pl_module.epoch_end_time = time.time()

        current_epoch = trainer.current_epoch

        # Stop the profiler
        if self.profileTest(current_epoch=current_epoch):
            pl_module.profiler.stop()
            print("Ended profiling")

        # Compute performance metrics
        epoch_train_time = pl_module.epoch_end_time - pl_module.epoch_start_time

        # num_samples = number of datapoints passed to a process (per epoch)
        # the sampler is likely a DistributedSampler because of DDP internals (I made this work for a custom DistributedSampler)
        num_samples = trainer.train_dataloader.batch_sampler.num_samples
        throughput = num_samples / epoch_train_time

        # Log the epoch training time and throughput
        # self.log("epoch_train_time", epoch_train_time)
        wandb.log(
            {f"epoch_train_time": epoch_train_time},
            step=trainer.global_step,
        )

        wandb.log(
            {f"epoch_throughput": throughput},
            step=trainer.global_step,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Called when a training batch ends.

        - If the profiler is active, step the profiler.

        Args:
            trainer: The PyTorch Lightning trainer object.
            pl_module: The PyTorch Lightning module being trained.
            outputs: The outputs of the training batch.
            batch: The current batch of data.
            batch_idx: The index of the current batch.

        Returns:
            None
        """
        current_epoch = trainer.current_epoch
        if self.profileTest(current_epoch=current_epoch):
            pl_module.profiler.step()

    def on_train_start(self, trainer, pl_module):
        """
        Called when the training starts.

        - Logs the start time of the training.

        Args:
            trainer: The PyTorch Lightning trainer object.
            pl_module: The PyTorch Lightning module being trained.
        """
        pl_module.train_start_time = time.time()

    def on_train_end(self, trainer, pl_module):
        """
        Callback function called at the end of the training process.

        - Logs a number of metrics (to files on net_scratch and optionally to wandb).

        Args:
            trainer: The PyTorch Lightning Trainer object.
            pl_module: The PyTorch Lightning module being trained.

        Returns:
            None
        """

        pl_module.train_end_time = time.time()
        pl_module.total_train_time = (
            pl_module.train_end_time - pl_module.train_start_time
        )
        wandb.log(
            {"total_train_time": pl_module.total_train_time}, step=trainer.global_step
        )

        print("Finished training. Now outputting performance metrics...")
        print(f"Total training time: {pl_module.total_train_time}")
        print("Finished training")
        wandb.finish()
        print("Closed the wandb logger")
