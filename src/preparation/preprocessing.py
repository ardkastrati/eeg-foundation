import mne
import numpy as np

from pyprep.utils import _eeglab_create_highpass, _eeglab_fir_filter
from pyprep.prep_pipeline import PrepPipeline
from meegkit import dss
from pyprep.utils import _eeglab_interpolate_bads

def create_raw(data, events):
    # Now we deal with MNE to get the right data structure ...
    # Create some dummy metadata
    channels = data.columns.tolist()
    sampling_freq = int(events["Rate"].iloc[0])
    ch_types = ["eeg" for ch_name in channels]
    info = mne.create_info(channels, ch_types=ch_types, sfreq=sampling_freq)
    
    eeg_data = np.array(data, dtype='float').T / 1000000 # in Volt
    raw = mne.io.RawArray(eeg_data, info)
    return raw


def Union(lst1, lst2):
    final_list = list(set(lst1) | set(lst2))
    return final_list
    
def find_bad_by_prep(raw, montage_kind, line_freq, matlab_strict=False):
    montage = mne.channels.make_standard_montage(montage_kind)
    
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(line_freq, raw.info['sfreq'] / 2, line_freq),
    }
    
    prep = PrepPipeline(raw.copy(), prep_params, montage, matlab_strict=matlab_strict)
    try:
        prep.fit()
    except OSError as e:
        print(f"An error occurred: {e}")
        raw.set_montage(montage)
        return raw, None
    
    print("BAD CHANNELS: ", Union(prep.interpolated_channels, prep.still_noisy_channels))
    raw.set_montage(montage)
    raw.info["bads"].extend(Union(prep.interpolated_channels, prep.still_noisy_channels))
    return raw, str(Union(prep.interpolated_channels, prep.still_noisy_channels))


def filter(raw, matlab_strict=False):
    EEG = raw.get_data()
    picks = mne.pick_types(raw.info, eeg=True)
    if matlab_strict:
        filt = _eeglab_create_highpass(0.1, raw.info['sfreq'])
        EEG[picks, :] = _eeglab_fir_filter(EEG[picks, :], filt)
    else:
        EEG = mne.filter.filter_data(
            EEG,
            sfreq=raw.info['sfreq'],
            l_freq=0.1,
            h_freq=None,
            picks=picks,
        )
    filtered_raw = mne.io.RawArray(EEG, raw.info)
    return filtered_raw



def zapline_clean(raw, fline=50):
    EEG = raw.get_data() # Convert mne data to numpy darray
    picks = mne.pick_types(raw.info, eeg=True)
    sfreq = raw.info['sfreq'] # Extract the sampling freq
       
    # Apply MEEGkit toolbox function
    out , _ = dss.dss_line(EEG[picks, :].T, fline, sfreq, nkeep=1) # fline (Line noise freq) = 50 Hz for Europe
    EEG[picks, :] = out.T
    cleaned_raw = mne.io.RawArray(EEG, raw.info) # Convert output to mne RawArray again

    return cleaned_raw


def interpolate(raw, matlab_strict=False):
    if matlab_strict:
        _eeglab_interpolate_bads(raw)
    else:
        raw.interpolate_bads()
    return raw

def average_reference(raw):
    avg = raw.copy().add_reference_channels(ref_channels="AVG_REF")
    avg = avg.set_eeg_reference(ref_channels="average")
    return avg