import math
import numpy as np

import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


class DurationBasedDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        id_to_channel_to_signals,
        max_durations,
        max_duration=35,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
    ):
        super().__init__(dataset=dataset)
        self.id_to_channel_to_signals = id_to_channel_to_signals
        self.max_durations = max_durations
        self.max_duration = max_duration
        self.shuffle = shuffle
        self.seed = seed

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.batch_indices = []
        self.generate_batches()

    def generate_batches(self):
        self.batch_indices = []

        for subject_id, channel_to_signals in self.id_to_channel_to_signals.items():
            for channel, signals in channel_to_signals.items():
                current_batch = []
                current_duration = 0

                for signal in signals:
                    idx, index_element = signal
                    sr = index_element["sr"]
                    time_used = 0
                    unused_duration = index_element["duration"] - time_used

                    while unused_duration > 0:
                        # Calculate the duration that can be used in the current batch
                        newly_used_duration = min(
                            unused_duration, self.max_duration - current_duration
                        )

                        # Add the signal to the batch
                        current_batch.append((idx, sr, newly_used_duration, time_used))

                        # Update the time_used and current_duration
                        time_used += newly_used_duration
                        current_duration += newly_used_duration

                        # Update the unused_duration
                        unused_duration -= newly_used_duration

                        # If current duration exceeds max duration, finalize the batch and start a new one
                        if current_duration >= self.max_duration:
                            self.batch_indices.append(current_batch)
                            current_batch = []
                            current_duration = 0

                # If there is any remaining batch, add it to the list of batches
                if current_batch:
                    self.batch_indices.append(current_batch)

        self.total_size = len(self.batch_indices)
        self.num_samples = math.ceil(self.total_size / self.num_replicas)

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.total_size, generator=g).tolist()
        else:
            indices = list(range(self.total_size))

        indices += indices[: (self.num_samples * self.num_replicas - len(indices))]
        assert len(indices) == self.num_samples * self.num_replicas

        # Subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        batch_indices = [self.batch_indices[i] for i in indices]

        return iter(batch_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
