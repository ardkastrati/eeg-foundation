import src.models.mae_original as mae
from src.data.transforms import crop_spectrogram, load_channel_data, fft_256
import torch
import mne
import lightning
import matplotlib.pyplot as plt
import json
import numpy as np
import tempfile
import os
import time
from socket import gethostname


def load_and_save_spgs(
    raw_paths,
    STORDIR="/scratch/mae",
    TMPDIR="/tmp",
    window_size=2.0,
    window_shift=0.125,
    target_size=(64, 2048),
):
    """
    Stores the spectrograms using the NUMPY library. You can chose where to store them.
    It divides them into directories of 1000 files each, I ran into problems with storing 200'000+ files in one directory.

    I have used /scratch (the local storage of the computing node) and if you have enough memory available /dev/shm is an option (basicly just a filesystem
    mounted on memory, so might be faster)
    The TMPDIR is used to store the indexing into the saved files so that all processes have easy access to it.
    """

    print("using temporary directory   : " + TMPDIR)
    print("storing in directory: " + STORDIR)

    spg_index = {}
    n_samples = len(raw_paths)
    n_directories = (n_samples // 1000) + 1
    c_loader = load_channel_data(precrop=True, crop_idx=[60, 316])
    fft = fft_256(window_size=window_size, window_shift=window_shift, cuda=True)
    crop = crop_spectrogram(target_size=target_size)

    # parent = tempfile.TemporaryDirectory(dir=STORDIR, delete=False)
    if not os.path.exists(STORDIR):
        os.makedirs(
            STORDIR
        )  # This will create the directory if it does not exist, given proper permissions.
    elif not os.access(STORDIR, os.W_OK):
        print(f"The directory {STORDIR} is not writable. Please check the permissions.")

    parent = tempfile.TemporaryDirectory(dir=STORDIR)
    subdir = {}
    data_index = {}
    for i in range(n_directories):
        subdir[i] = tempfile.mkdtemp(dir=parent.name)

    print(subdir)
    start = time.time()

    for ind, raw in enumerate(raw_paths):

        if ind % 1000 == 0:
            print("now at")
            print(ind)

        signal = c_loader(raw)

        signal = signal.to("cuda")
        spg = fft(signal)

        spg = crop(spg)
        spg = spg.cpu().numpy()

        subdir_index = ind // 1000
        index = ind % 1000
        file_index = "spg" + str(index) + ".npy"
        save_path = os.path.join(subdir[subdir_index], file_index)

        np.save(save_path, spg)
        data_index[ind] = save_path

    end = time.time()
    print(end - start)

    print("Saved the spectrograms locally")
    index_path = os.path.join(parent.name, "data_index")
    with open(index_path, "w") as file:
        json.dump(data_index, file)

    with open(os.path.join(TMPDIR, f"index_path_{gethostname()}.txt"), "w") as file:
        file.write(index_path)

    return data_index, parent, subdir
