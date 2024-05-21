import numpy as np
from torch.utils.data import Sampler


class DurationBasedSampler(Sampler):
    def __init__(self, id_to_channel_to_signals, max_durations, max_duration=33):
        super().__init__()
        self.id_to_channel_to_signals = id_to_channel_to_signals
        self.max_durations = max_durations
        self.max_duration = max_duration
        self.batch_indices = []
        self.generate_batches()
        # print(self.batch_indices)

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

    def __iter__(self):
        for batch in self.batch_indices:
            yield batch

    def __len__(self):
        return len(self.batch_indices)
