from datetime import datetime
import os
from socket import gethostname
from lightning import Callback
import torch
import wandb
import csv
from socket import gethostname
from pytorch_lightning.loggers import WandbLogger
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
        self.log_path = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M"))
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
        #     current_epoch < 20
        #     # current_epoch < 2
        #     # or (5 <= current_epoch and current_epoch < 10)
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
                schedule=torch.profiler.schedule(
                    wait=0, warmup=1, active=nr_batches - 1, repeat=1
                ),
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

        # Log losses
        # Log the losses for the current epoch
        # current_epoch = trainer.current_epoch
        # for global_step, epoch, batch_idx, loss in pl_module.train_losses[
        #     current_epoch
        # ]:
        #     wandb.log({f"train_loss": loss}, step=global_step)
        # might want to log with step=current_epoch...

        # Compute and log the average loss for the current epoch
        # if len(pl_module.train_losses[current_epoch]) > 0:
        #     losses = [loss for _, _, _, loss in pl_module.train_losses[current_epoch]]
        #     avg_loss = sum(losses) / len(losses)
        #     wandb.log(
        #         {f"average_train_loss": avg_loss},
        #         step=trainer.global_step,
        #         commit=False,
        #     )

        # Compute performance metrics
        epoch_train_time = pl_module.epoch_end_time - pl_module.epoch_start_time

        # num_samples = number of datapoints passed to a process (per epoch)
        # the sampler is likely a DistributedSampler because of DDP internals (I made this work for DistributedSampler)
        num_samples = trainer.train_dataloader.sampler.num_samples
        throughput = num_samples / epoch_train_time

        # Log the epoch training time and throughput
        # self.log("epoch_train_time", epoch_train_time)
        wandb.log(
            {f"epoch_train_time": epoch_train_time},
            step=trainer.global_step,
        )
        # self.log(
        #     name="epoch_train_time",
        #     value=epoch_train_time,
        #     on_epoch=True,
        #     rank_zero_only=False,
        # )

        pl_module.epoch_train_times.append(
            (trainer.global_step, current_epoch, epoch_train_time)
        )
        # self.log("epoch_throughput", throughput)
        wandb.log(
            {f"epoch_throughput": throughput},
            step=trainer.global_step,
        )
        # self.log(
        #     name="epoch_throughput",
        #     value=throughput,
        #     on_epoch=True,
        #     rank_zero_only=False,
        # )
        pl_module.epoch_train_throughputs.append(
            (trainer.global_step, current_epoch, throughput)
        )

        # wandb.log(
        #     {"epoch": self.current_epoch}, commit=True
        # )  # This commits the accumulated metrics

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
        print("Finished training. Now outputting performance metrics...")

        ########################################
        # Write total_train_time to a file
        # with open(
        #     f"{self.run_dir}/metrics/total_train_time_{trainer.global_rank}_{pl_module.hostname}.txt",
        #     "w",
        # ) as f:
        #     f.write(str(pl_module.total_train_time))
        # trainer.logger.experiment.log({"total_train_time": pl_module.total_train_time})
        # wandb.log(
        #     {f"total_train_time": pl_module.total_train_time},
        #     step=trainer.global_step,
        # )
        # Also log the total_train_time to wandb
        # if trainer.logger is not None:
        #     trainer.logger.log_metrics(
        #         {"total_train_time": pl_module.total_train_time},
        #         step=trainer.global_step,
        #     )
        ########################################
        # Write epoch_train_times to a CSV file
        # with open(
        #     f"{self.run_dir}/metrics/train_times_{trainer.global_rank}_{pl_module.hostname}.csv",
        #     "w",
        #     newline="",
        # ) as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["Global_Step", "Epoch", "Time"])
        #     writer.writerows(pl_module.epoch_train_times)
        # Also log the epoch_train_times to wandb
        # if trainer.logger is not None:
        #     for global_step, _, t in pl_module.epoch_train_times:
        #         trainer.logger.log_metrics({"epoch_train_time": t}, step=global_step)
        ########################################
        # Write epoch_train_throughputs to a CSV file
        # with open(
        #     f"{self.run_dir}/metrics/train_throughputs_{trainer.global_rank}_{pl_module.hostname}.csv",
        #     "w",
        #     newline="",
        # ) as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["Global_Step", "Epoch", "Throughput"])
        #     writer.writerows(pl_module.epoch_train_throughputs)
        # Also log the epoch_train_throughputs to wandb
        # if trainer.logger is not None:
        #     for global_step, _, throughput in pl_module.epoch_train_throughputs:
        #         trainer.logger.log_metrics(
        #             {"epoch_throughput": throughput}, step=global_step
        #         )
        ########################################
        # Log the train_losses
        # with open(
        #     f"{self.run_dir}/metrics/train_losses_{trainer.global_rank}_{pl_module.hostname}.csv",
        #     "w",
        #     newline="",
        # ) as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["Global_Step", "Epoch", "Batch_Idx", "Loss"])
        #     writer.writerows(pl_module.train_losses)

        # Adjusted code for train_losses
        # with open(
        #     f"{self.run_dir}/metrics/train_losses_{trainer.global_rank}_{pl_module.hostname}.csv",
        #     "w",
        #     newline="",
        # ) as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["Global_Step", "Epoch", "Batch_Idx", "Loss"])
        #     for _, epoch_losses in enumerate(pl_module.train_losses):
        #         for global_step, epoch, batch_idx, loss in epoch_losses:
        #             writer.writerow([global_step, epoch, batch_idx, loss])
        # Also log the train_losses to wandb
        # if trainer.logger is not None:
        #     for global_step, _, _, loss in pl_module.train_losses:
        #         trainer.logger.log_metrics({"train_loss": loss}, step=global_step)
        ########################################
        # Log the val_losses
        # with open(
        #     f"{self.run_dir}/metrics/val_losses_{trainer.global_rank}_{pl_module.hostname}.csv",
        #     "w",
        #     newline="",
        # ) as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["Global_Step", "Epoch", "Batch_Idx", "Loss"])
        #     writer.writerows(pl_module.val_losses)
        # with open(
        #     f"{self.run_dir}/metrics/val_losses_{trainer.global_rank}_{pl_module.hostname}.csv",
        #     "w",
        #     newline="",
        # ) as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["Global_Step", "Epoch", "Batch_Idx", "Loss"])
        #     for _, epoch_losses in enumerate(pl_module.val_losses):
        #         for global_step, epoch, batch_idx, loss in epoch_losses:
        #             writer.writerow([global_step, epoch, batch_idx, loss])
        # Also log the val_losses to wandb
        # if trainer.logger is not None:
        #     for global_step, _, _, loss in pl_module.val_losses:
        #         trainer.logger.log_metrics({"val_loss": loss}, step=global_step)
        ########################################
        # if trainer.logger is not None:
        #     trainer.logger.finalize("success")
        print("Finished training")
        wandb.finish()
        print("Closed the wandb logger")
