import sys
import numpy as np
import torch
from torch.utils.data import Dataset


# == Define Datasets ==


class RawDataset(Dataset):
    def __init__(self, channel_index):
        self.channel_index = channel_index

    def __getitem__(self, info):
        idx, start, dur = info
        channel_info = self.channel_index[idx]
        signal = channel_info["signal"]
        signal = signal[
            int(start * channel_info["sr"]) : int((start + dur) * channel_info["sr"])
        ]
        return {
            "signal": torch.tensor(signal),
            "path": channel_info["path"],
            "channel": channel_info["channel"],
            "sr": channel_info["sr"],
            "dur": channel_info["dur"],
        }


class ChannelDataset(Dataset):
    def __init__(self, channel_index):
        self.channel_index = channel_index

    def __getitem__(self, info):
        idx, start, dur = info
        channel_info = self.channel_index[idx]
        signal = np.load(channel_info["path"])
        signal = signal[
            int(start * channel_info["sr"]) : int((start + dur) * channel_info["sr"])
        ]
        return {
            "signal": torch.tensor(signal),
            "path": channel_info["path"],
            "channel": channel_info["channel"],
            "sr": channel_info["sr"],
            "dur": channel_info["dur"],
        }

    def __len__(self):
        return len(self.channel_index)


class TrialDataset(Dataset):
    def __init__(self, trial_index):
        self.trial_index = trial_index
        self.win_sizes = [-1] * len(trial_index)

    def __getitem__(self, trial_idx):
        trial_info = self.trial_index[trial_idx]
        win_size = self.win_sizes[trial_idx]

        channels = []
        signals = []

        for chn, path in zip(trial_info["channels"], trial_info["paths"]):
            signal = np.load(path)
            signal = torch.tensor(signal)
            signals.append(signal)
            channels.append(chn)

        return (signals, channels, win_size, trial_info["sr"], trial_info["dur"])

    def __len__(self):
        return len(self.trial_index)


class PathDataset(Dataset):
    def __init__(self, signal_index):
        self.signal_index = signal_index
        self.paths = [index_element["path"] for index_element in signal_index.values()]
        self.channels = [
            index_element["channel"] for index_element in signal_index.values()
        ]
        self.len = sum(
            [index_element["duration"] for index_element in signal_index.values()]
        )

    def __getitem__(self, info):
        # print(info)
        idx, sr, dur, time_used = info
        # print(dur)
        signal_path = self.paths[idx]
        signal = np.load(signal_path)
        start_sample = int(sr * time_used)
        end_sample = start_sample + int(sr * dur)
        signal_chunk = signal[start_sample:end_sample]
        chn = self.channels[idx]
        return (torch.tensor(signal_chunk), chn, sr, dur)

    def __len__(self):
        return self.len
