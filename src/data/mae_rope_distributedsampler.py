import math
import sys
import time
import numpy as np

import torch
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


class ByTrialDistributedSampler(DistributedSampler):
    def __init__(
        self,
        mode,
        full_dataset,
        subset_indices,
        drop_last=False,
        patch_size=16,
        max_nr_patches=8_500,
        win_shifts=[0.25, 0.5, 1, 2, 4, 8],
        win_shift_factor=0.25,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
    ):
        super().__init__(dataset=full_dataset, drop_last=drop_last)

        self.mode = mode

        self.full_dataset = full_dataset
        self.subset_indices = subset_indices

        self.drop_last = drop_last

        # Group channels in subset_indices by (subject, trial)
        id_to_sr_to_trial_to_channels = {}
        for idx, channel_idx in enumerate(self.subset_indices):
            channel_info = self.full_dataset.channel_index[channel_idx]
            subject_id = channel_info["SubjectID"]
            sr = channel_info["sr"]
            trial_idx = channel_info["trial_idx"]
            if subject_id not in id_to_sr_to_trial_to_channels:
                id_to_sr_to_trial_to_channels[subject_id] = {sr: {trial_idx: [idx]}}
            elif sr not in id_to_sr_to_trial_to_channels[subject_id]:
                id_to_sr_to_trial_to_channels[subject_id][sr] = {trial_idx: [idx]}
            elif trial_idx not in id_to_sr_to_trial_to_channels[subject_id][sr]:
                id_to_sr_to_trial_to_channels[subject_id][sr][trial_idx] = [idx]
            else:
                id_to_sr_to_trial_to_channels[subject_id][sr][trial_idx].append(idx)
        self.id_to_sr_to_trial_to_channels = id_to_sr_to_trial_to_channels

        print(f"[Sampler] # {mode}-Subjects: {len(id_to_sr_to_trial_to_channels)}")

        self.patch_size = patch_size
        self.max_nr_patches = max_nr_patches
        self.win_shifts = win_shifts
        self.win_shift_factor = win_shift_factor

        self.num_replicas = (
            num_replicas if num_replicas is not None else dist.get_world_size()
        )
        self.rank = rank if rank is not None else dist.get_rank()

        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        start_time = time.time()
        self.generate_batches()
        print(
            f"[Sampler] # {mode}-GenerateBatches took: {round(time.time() - start_time,2)}s",
            file=sys.stderr,
        )

    def get_nr_y_patches(self, win_size, sr):
        return int((sr / 2 * win_size + 1) / self.patch_size)

    def get_nr_x_patches(self, win_size, dur):
        win_shift = win_size * self.win_shift_factor
        x_datapoints_per_second = 1 / win_shift
        x_datapoints = dur * x_datapoints_per_second + 1
        return int(x_datapoints / self.patch_size)

    def get_nr_patches(self, win_size, sr, dur):
        # to be cautious: we add 1 to the number of patches in both directions
        # return (self.get_nr_y_patches(win_size=win_size, sr=sr) + 1) * (
        #     self.get_nr_x_patches(win_size=win_size, dur=dur) + 1
        # )
        return self.get_nr_y_patches(win_size=win_size, sr=sr) * (
            self.get_nr_x_patches(win_size=win_size, dur=dur)
        )

    def get_max_nr_patches(self, sr, dur):
        suitable_win_sizes = self.get_suitable_win_sizes(sr, dur)
        return max(
            [self.get_nr_patches(win_size, sr, dur) for win_size in suitable_win_sizes]
        )

    def get_suitable_win_sizes(self, sr, dur):
        return [
            win_shift
            for win_shift in self.win_shifts
            if self.get_nr_y_patches(win_shift, sr) >= 1
            and self.get_nr_x_patches(win_shift, dur) >= 1
        ]

    def get_suitable_win_size(self, sr, dur):
        win_sizes = self.get_suitable_win_sizes(sr, dur)
        return None if not win_sizes else win_sizes[0]

    def get_max_y_patches(self, sr, dur):
        max_win_shift = max(self.get_suitable_win_sizes(sr, dur))
        return self.get_nr_y_patches(win_size=max_win_shift, sr=sr)

    def generate_batches(self):
        self.batch_indices = []

        for (
            subject_id,
            sr_to_trial_to_channels,
        ) in self.id_to_sr_to_trial_to_channels.items():

            for sr, trial_to_channels in sr_to_trial_to_channels.items():

                current_batch = []
                current_nr_patches = 0
                min_batch_dur = float("inf")

                for trial_idx, channels in trial_to_channels.items():

                    for idx in channels:

                        channel_idx = self.subset_indices[idx]
                        channel_info = self.full_dataset.channel_index[channel_idx]

                        new_patches = self.get_max_nr_patches(
                            sr=channel_info["sr"],
                            dur=channel_info["dur"],
                        )
                        assert (
                            channel_info["sr"] == sr
                        ), f"channel_info['sr'] ({channel_info['sr']}) != sr ({sr})"
                        assert (
                            new_patches != 0
                        ), f"new_patches == 0, sr={channel_info['sr']}, dur={channel_info['dur']}"

                        # Also add the column of separation patches, be pessimistic
                        #  (we need to be pessimistic because we don't have a signal duration for just one patch,
                        #  so we get a different value for different win_shifts)
                        sep_patches = self.get_max_y_patches(
                            channel_info["sr"], min(min_batch_dur, channel_info["dur"])
                        )

                        # If adding the new patches would exceed the maximum number of patches,
                        #  finalize this batch and start a new one
                        if current_nr_patches <= self.max_nr_patches and (
                            current_nr_patches + sep_patches + new_patches
                            > self.max_nr_patches
                            or current_nr_patches + sep_patches > self.max_nr_patches
                        ):
                            if current_batch:
                                assert (
                                    current_nr_patches <= self.max_nr_patches
                                ), f"1: current_nr_patches={current_nr_patches} > max_nr_patches={self.max_nr_patches}"
                                self.batch_indices.append(current_batch)
                            current_batch = []
                            current_nr_patches = 0
                            min_batch_dur = float("inf")

                        current_nr_patches += new_patches
                        if current_batch:
                            current_nr_patches += sep_patches
                        current_batch.append(idx)
                        min_batch_dur = min(min_batch_dur, channel_info["dur"])

                if current_batch:
                    assert (
                        current_nr_patches <= self.max_nr_patches
                    ), f"2: current_nr_patches={current_nr_patches} > max_nr_patches={self.max_nr_patches}"
                    self.batch_indices.append(current_batch)

        self.total_size = len(self.batch_indices)
        self.num_samples = math.ceil(self.total_size / self.num_replicas)

    def generate_batches_old(self):
        self.batch_indices = []

        for subject_id, trials in self.id_to_trials.items():

            current_batch = []
            current_nr_patches = 0

            # TODO: have maximum number of channels per trial
            # split trial["channels"] into sublists, %1, %2... %10
            # TODO: have maximum duration for these subtrials, split longer signals

            for idx in trials:

                # TODO: custom logic! => move to collate_fn
                # - try with randomly uniform sampling from [.25, .5, 1, 2, 4, 8]
                # - coordinate with Ard
                # - let's do this per-batch for now
                # TODO: also, this probably needs to be recomputed in each epoch
                win_size = 1

                trial_idx = self.subset_indices[idx]
                trial_info = self.full_dataset.trial_index[trial_idx]

                new_patches = self.get_nr_patches(
                    n_channels=len(trial_info["channels"]),
                    win_size=win_size,
                    win_shift=win_size * self.win_shift_factor,
                    sr=trial_info["sr"],
                    dur=trial_info["dur"],
                )

                if current_nr_patches + new_patches > self.max_nr_patches:
                    # TODO: what to do when signal is bigger than max_nr_patches?
                    if current_batch:
                        self.batch_indices.append(current_batch)
                    win_size = 1  # TODO: custom logic!
                    current_batch = []
                    current_nr_patches = 0

                self.full_dataset.win_sizes[trial_idx] = win_size
                current_batch.append(idx)
                current_nr_patches += new_patches

            if current_batch:
                self.batch_indices.append(current_batch)

        self.total_size = len(self.batch_indices)
        self.num_samples = math.ceil(self.total_size / self.num_replicas)

    def __iter__NEW(self):
        # Recompute batch each epoch
        pass

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.batch_indices), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.batch_indices)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = 0
            while (self.total_size + padding_size) % self.num_replicas != 0:
                padding_size += 1
            final_size = self.total_size + padding_size
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            final_size = self.total_size // self.num_replicas * self.num_replicas
            indices = indices[:final_size]
        assert len(indices) % self.num_replicas == 0

        # subsample
        indices = indices[self.rank : final_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # This is what changed compared to the DistributedSampler super-class
        batch_indices = [self.batch_indices[i] for i in indices]

        return iter(batch_indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
