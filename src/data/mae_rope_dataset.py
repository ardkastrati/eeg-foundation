import numpy as np
import torch
from torch.utils.data import Dataset


# == Initialize Datasets ==
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
        print(info)
        idx, sr, dur, time_used = info
        # print(dur)
        signal_path = self.paths[idx]
        signal = np.load(signal_path)
        start_sample = int(sr * time_used)
        end_sample = start_sample + int(sr * dur)
        signal_chunk = signal[start_sample:end_sample]
        chn = self.channels[idx]
        return (torch.tensor(signal_chunk), sr, chn)

    def __len__(self):
        return self.len
