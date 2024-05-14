import numpy as np
import mne


# ==============================================================================================================================


def create_raw(
    data,
    ch_names1,
    sr,
    ch_names2=None,
):
    if ch_names2 == None:
        ch_names2 = ch_names1
    ch_types = ["eeg" for _ in range(len(ch_names1))]
    info = mne.create_info(ch_names2, ch_types=ch_types, sfreq=sr)
    eeg_data = (
        np.array(data[ch_names1].T, dtype="float") / 1_000_000
    )  # in Volt # TODO not sure if each dataset is in uv
    raw = mne.io.RawArray(eeg_data, info)
    return raw


# ==============================================================================================================================

from pyprep.prep_pipeline import PrepPipeline


def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list


def find_bad_by_prep(
    raw, montage_kind, grid_frequency, matlab_strict=False, hacky=False
):

    # Add a montage to the data
    montage = mne.channels.make_standard_montage(montage_kind)

    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(grid_frequency, raw.info["sfreq"] / 2, grid_frequency),
    }

    prep = PrepPipeline(raw.copy(), prep_params, montage, matlab_strict=matlab_strict)

    prep.fit()
    raw.info["bads"].extend(
        Union(prep.interpolated_channels, prep.still_noisy_channels)
    )

    # print("BAD CHANNELS: ", Union(prep.interpolated_channels, prep.still_noisy_channels))
    raw.set_montage(montage)

    return raw
