from datetime import datetime
import os
from socket import gethostname
from lightning import Callback
import torch
import time
import wandb


class ProfilerCallback(Callback):

    def __init__(
        self,
        log_dir,
        record_shapes,
        with_stack,
        profile_memory,
        log_epoch_freq,
        hostname,
    ):
        super().__init__()
        self.log_dir = log_dir
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.profile_memory = profile_memory
        self.epoch_freq = log_epoch_freq
        self.log_path = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d_%H-%M"))
        os.makedirs(self.log_path, exist_ok=True)
        self.hostname = hostname

    def profileTest(self, current_epoch: int):
        """Check whether profiler should be active for this epoch.

        Args:
            current_epoch (int): The current epoch number.

        Returns:
            bool: True if the test should be profiled, False otherwise.
        """
        if (
            current_epoch < 2
            or (5 <= current_epoch and current_epoch < 10)
            or current_epoch % self.epoch_freq == 0
        ):
            return True
        else:
            return False

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

        if self.profileTest(current_epoch=current_epoch):
            pl_module.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=1, warmup=1, active=nr_batches - 2, repeat=1
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

        print("Starting train epoch")
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
        epoch_train_time = pl_module.epoch_end_time - pl_module.epoch_start_time

        print("Ending train epoch")

        # Handle profiler
        current_epoch = trainer.current_epoch
        if self.profileTest(current_epoch=current_epoch):
            pl_module.profiler.stop()
            print("Ending profiling")
            del pl_module.profiler  # not sure if necessary
        # pl_module.epoch_train_times.extend(epoch_train_time)
        # wandb.log(
        #     {
        #         "epoch_nr": current_epoch,
        #         "epoch_train_time": epoch_train_time,
        #         "hostname": self.hostname,
        #         "global_rank": trainer.global_rank,
        #     }
        # )
        self.log("epoch_nr", current_epoch)
        self.log("epoch_train_time", epoch_train_time)
        self.log("hostname", int(gethostname()[-2:]))
        self.log("global_rank", trainer.global_rank)
        print(f"epoch_train_time: {epoch_train_time}")

        # Handle throughput metrics
        # dataset = (
        #     trainer.train_dataloader.dataset.dataset
        # )  # Access the SimpleDataset instance of this process

        # print("trying to access sampler")
        sampler = trainer.train_dataloader.sampler
        num_samples = (
            sampler.num_samples
        )  # nr of datapoints passed to a process (per epoch)
        print("num_samples", num_samples)

        throughput = num_samples / epoch_train_time
        print(
            f"throughput={throughput} on node {trainer.global_rank} on {gethostname()} in epoch {current_epoch}"
        )
        # Log the metrics
        # wandb.log(
        #     {
        #         "throughput": throughput,
        #         "epoch_nr": current_epoch,
        #         "hostname": self.hostname,
        #         "global_rank": trainer.global_rank,
        #     }
        # )
        self.log("throughput", throughput)
        self.log("epoch_nr", current_epoch)
        self.log("hostname", int(gethostname()[-2:]))
        self.log("global_rank", trainer.global_rank)
        # total_load_time = sum(dataset.load_times)
        # total_data_points = len(dataset) * trainer.train_dataloader.batch_size
        # throughput = total_data_points / total_load_time

        # Clear the load times for the next epoch
        # dataset.load_times = []

    # def training_epoch_end(self, outputs):
    #     print("training_epoch_end", self.hostname)
    #     total_load_time = sum(self.train_dataset.load_times)
    #     total_data_points = len(self.train_dataset) * self.batch_size
    #     throughput = total_data_points / total_load_time
    #     # Log the metrics
    #     self.log(
    #         {
    #             "total_load_time": total_load_time,
    #             "total_data_points": total_data_points,
    #             "throughput": throughput,
    #             "epoch": self.current_epoch,
    #             "hostname": self.hostname,
    #         }
    #     )
    #     # Clear the load times for the next epoch
    #     self.train_dataset.load_times = []

    # def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
    #     if trainer.current_epoch == 0 and batch_idx == 0:
    #         pl_module.train_start_time = time.time()

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

        - Logs the total training time.

        Args:
            trainer: The PyTorch Lightning Trainer object.
            pl_module: The PyTorch Lightning module being trained.

        Returns:
            None
        """
        pl_module.train_end_time = time.time()
        total_runtime = pl_module.train_end_time - pl_module.train_start_time
        # wandb.log(
        #     {
        #         "total_traintime": total_runtime,
        #         "hostname": self.hostname,
        #         "global_rank": trainer.global_rank,
        #     }
        # )
        # TODO: handle the logging later
        # self.logger.experiment.log("total_traintime", total_runtime)
        # self.logger.experiment.log("hostname", int(gethostname()[-2:]))
        # self.logger.experiment.log("global_rank", trainer.global_rank)
