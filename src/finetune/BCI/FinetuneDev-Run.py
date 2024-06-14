import json
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.preprocessing import StandardScaler
import os
import pickle
from tqdm import tqdm
import lightning as L
import torch.nn as nn
from torch.utils.data import random_split
from lightning.pytorch.callbacks import ModelCheckpoint
import sys
import random
from collections import Counter
from collections import defaultdict

import os
import numpy as np
import mne
import torch
from tqdm import tqdm
from mne.preprocessing import Xdawn
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import balanced_accuracy_score, mean_squared_error

sys.path.append("/home/maxihuber/eeg-foundation/")
L.seed_everything(42)

with open(
    "/home/maxihuber/eeg-foundation/src/data/components/channels_to_id.json", "r"
) as f:
    pkl_channels = set(json.load(f).keys())
    cli_channels = set(
        [
            "AF3",
            "AF4",
            "AF7",
            "AF8",
            "AFz",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "CP1",
            "CP2",
            "CP3",
            "CP4",
            "CP5",
            "CP6",
            "CPz",
            "Cz",
            "F1",
            "F2",
            "F3",
            "F4",
            "F5",
            "F6",
            "F7",
            "F8",
            "FC1",
            "FC2",
            "FC3",
            "FC4",
            "FC5",
            "FC6",
            "FCz",
            "FT7",
            "FT8",
            "Fp1",
            "Fp2",
            "Fz",
            "Mastoids",
            "O1",
            "O2",
            "Oz",
            "P1",
            "P2",
            "P3",
            "P4",
            "P5",
            "P6",
            "P7",
            "P8",
            "PO3",
            "PO4",
            "PO7",
            "PO8",
            "POz",
            "Pz",
            "T7",
            "T8",
            "TP7",
            "TP8",
            "Veog",
            "X",
            "Y",
            "Z",
        ]
    )
task_channels = pkl_channels | cli_channels


########################################################################################################################
# Clinical JSONs

cli_class = {
    "class_name": "Clinical",
    "time_col": "Time in Seconds",
    "prefix_filepath": "/itet-stor/maxihuber/deepeye_storage/foundation_clinical_prepared/",
    "load_mode": 0,
}

age = {
    "task_name": "Age",
    "task_type": "Regression",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_clinical_tasks/age.json",
    "out_dim": 1,
}

depression = {
    "task_name": "Depression",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_clinical_tasks/cli_depression.json",
    "out_dim": 2,
}

parkinsons = {
    "task_name": "Parkinsons",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_clinical_tasks/cli_parkinsons.json",
    "out_dim": 2,
}

schizophrenia = {
    "task_name": "Schizophrenia",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_clinical_tasks/cli_schizophrenia.json",
    "out_dim": 2,
}

sex = {
    "task_name": "Sex",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_clinical_tasks/sex.json",
    "out_dim": 2,
}


########################################################################################################################
# Motor-Imagery JSONs

mi_class = {
    "class_name": "Motor Imagery",
    "time_col": "time in seconds",
    "prefix_filepath": "/itet-stor/maxihuber/deepeye_storage/foundation_prepared/",
    "load_mode": 0,
}

eye_open_closed = {
    "task_name": "EyeOpenClosed",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/eye_open_closed.json",
    "out_dim": 2,
    "outputs": set(["eye open", "eye closed"]),
    "short_mode": False,
}

eye_vh = {
    "task_name": "EyeVH",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/eye_vh.json",
    "out_dim": 2,
    "outputs": set(["vertical", "horizontal"]),
    "short_mode": False,
}

flexion_extension_imaginary = {
    "task_name": "FlexionExtensionImaginary",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/flexion_extension_imaginary.json",
    "out_dim": 2,
    "outputs": set(
        [
            "hand movement imagined elbow flexion",
            "hand movement imagined elbow extension",
        ]
    ),
    "short_mode": False,
}

flexion_extension_real = {
    "task_name": "FlexionExtensionReal",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/flexion_extension_real.json",
    "out_dim": 2,
    "outputs": set(["hand movement elbow extension", "hand movement elbow flexion"]),
    "short_mode": False,
}

grasp_imaginary = {
    "task_name": "GraspImaginary",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/grasp_imaginary.json",
    "out_dim": 2,
    "outputs": set(["imagined palmar grasp", "imagined lateral grasp"]),
    "short_mode": False,
}

grasp_real = {
    "task_name": "GraspReal",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/grasp_real.json",
    "out_dim": 2,
    "outputs": set(["movement palmar grasp", "movement lateral grasp"]),
    "short_mode": False,
}

lr_imaginary = {
    "task_name": "LRImaginary",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/lr_imaginary.json",
    "out_dim": 2,
    "outputs": set(["left hand imagined movement", "right hand imagined movement"]),
    "short_mode": True,
}

lr_real = {
    "task_name": "LRReal",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/lr_real.json",
    "out_dim": 2,
    "outputs": set(["right hand movement", "left hand movement"]),
    "short_mode": True,
}

mi_task_body_parts_real = {
    "task_name": "BodyPartsReal",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/mi_task_body_parts.json",
    "out_dim": 4,
    "outputs": set(
        ["rest", "right hand movement", "foot movement", "left hand movement"]
    ),
    "short_mode": True,
}

mi_task_body_parts_imagined = {
    "task_name": "BodyPartsImagined",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/mi_task_body_parts.json",
    "out_dim": 4,
    "outputs": set(
        [
            "rest",
            "right hand imagined movement",
            "foot imagined movement",
            "left hand imagined movement",
            "tongue imagined movement",
        ]
    ),
    "short_mode": True,
}

pronation_supination_real = {
    "task_name": "PronationSupinationReal",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/pronation_supination_real.json",
    "out_dim": 2,
    "outputs": set(["movement supination", "movement pronation"]),
    "short_mode": False,
}

pronation_supination_imaginary = {
    "task_name": "PronationSupinationImaginary",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/mi/pronation_supination_imaginary.json",
    "out_dim": 2,
    "outputs": set(["imagined supination", "imagined pronation"]),
    "short_mode": False,
}

########################################################################################################################
# ERP JSONs

erp_class = {
    "class_name": "Error-Related Potential",
    "time_col": "time in seconds",
    "prefix_filepath": "/itet-stor/maxihuber/deepeye_storage/foundation_prepared/",
    "load_mode": 0,
}

erp = {
    "task_name": "ERP",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/erp/erp_all.json",
    "out_dim": 5,
    "outputs": set(
        [
            "Participant is in resting state",
            "with event-related potential",
            "Participant is in interval between two flashes",
            "without event-related potential",
            "Participant keeps closing eyes",
        ]
    ),
}

errp = {
    "task_name": "ERRP",
    "task_type": "Classification",
    "json_path": "/itet-stor/maxihuber/deepeye_storage/foundation_tasks/erp/errp_all.json",
    "out_dim": 7,
    "outputs": set(
        [
            "Target is located in the right",
            "without error-related potential",
            "The cursor moves to the left",
            "The feedback consisted in the selected item is presented on the screen",
            "The cursor moves to the right",
            "with error-related potential",
            "Target is located in the left",
        ]
    ),
}

########################################################################################################################
# EyeNet JSONs

eye_class = {
    "class_name": "EyeNet",
    "time_col": "time",
    "prefix_filepath": "/itet-stor/maxihuber/deepeye_storage/foundation_prepared/",
    "load_mode": 1,
}

eye_dir_amp = {
    "task_name": "EyeNetDirectionAmp",
    "task_type": "Regression",
    "json_path": [
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Direction_Amp_train.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Direction_Amp_val.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Direction_Amp_test.json",
    ],
    "out_dim": 1,
}

eye_dir_ang = {
    "task_name": "EyeNetDirectionAng",
    "task_type": "Regression",
    "json_path": [
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Direction_Ang_train.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Direction_Ang_val.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Direction_Ang_test.json",
    ],
    "out_dim": 1,
}

eye_lr = {
    "task_name": "EyeNetLR",
    "task_type": "Classification",
    "json_path": [
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_LR_train.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_LR_val.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_LR_test.json",
    ],
    "out_dim": 2,
}

eye_position = {
    "task_name": "EyeNetPosition",
    "task_type": "Regression",
    "json_path": [
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Position_train.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Position_val.json",
        "/itet-stor/maxihuber/deepeye_storage/eegeyenet_tasks/EEGEyeNet_Position_test.json",
    ],
    "out_dim": 2,
}

########################################################################################################################
# Select the class and task

# used_class = cli_class
used_class = mi_class
# used_class = erp_class
# used_class = eye_class


# used_task = age
# used_task = depression
# used_task = parkinsons
# used_task = schizophrenia
# used_task = sex
#
# used_task = eye_open_closed
# used_task = eye_vh
# used_task = flexion_extension_imaginary
# used_task = flexion_extension_real
# used_task = grasp_real
used_task = lr_imaginary
# used_task = lr_real
# used_task = mi_task_body_parts_real
# used_task = mi_task_body_parts_imagined
# used_task = pronation_supination_real
# used_task = pronation_supination_imaginary
#
# used_task = erp
# used_task = errp
#
# used_task = eye_dir_amp
# used_task = eye_dir_ang
# used_task = eye_lr
# used_task = eye_position

class_name = used_class["class_name"]
time_col = used_class["time_col"]
prefix_filepath = used_class["prefix_filepath"]
load_mode = used_class["load_mode"]
task_name = used_task["task_name"]
task_type = used_task["task_type"]
json_path = used_task["json_path"]
out_dim = used_task["out_dim"]
short_mode = used_task["short_mode"] if "short_mode" in used_task else False

balance_datasets = True


def load_index0(data_index_path):
    with open(data_index_path, "r") as f:
        train_test_dict = json.load(f)
    train_samples = train_test_dict["train"]
    test_samples = train_test_dict["test"]
    return train_samples, test_samples


def load_index1(data_index_paths):
    all_samples = []
    for data_index_path in data_index_paths:
        with open(data_index_path, "r") as f:
            subset_dict = json.load(f)
        all_samples.append(list(subset_dict.values())[0])
    return all_samples[0], all_samples[1], all_samples[2]


dataset_dict = {
    "ERP_ERP_ANA": 0,
    "RS_RS_ALPHA": 1,
    "ERP_ERP_BISC": 2,
    "ERP_ERP_BBI": 3,
    "ERP_ERP_BICF": 4,
    "ERP_ERP_BICD": 5,
    "RS_RS_SPIS": 6,
    "MI_MI_HGD": 7,
    "MI_MI_SCP": 8,
    "ErrP_ErrP_MERP": 9,
    "MI_MI_ULM": 10,
    "MI_MI_VEP": 11,
    "MI_MI_LR": 12,
    "MI_BBCI_IV_Graz_b": 13,
    "MI_MI_EB": 14,
    "MI_BBCI_IV_Graz_a": 15,
    "MI_MI_GVH_V": 16,
    "MI_MI_GAL": 17,
    "MI_MI_Two": 18,
    "MI_MI_GVH_H": 19,
    "MI_MI_II": 20,
    "ErrP_ErrP_BCI": 21,
    "MI_MI_GVH_G": 22,
    "MI_MI_Limb": 23,
    "MI_MI_SCI": 24,
    "MI_BBCI_IV_Berlin": 25,
    "MI_eegmmidb": 26,
    "ERP_ERP_FHD": 27,
    "RS_RS_EID": 28,
}


def extract_dataset_name(file_path, dataset_dict):
    for name in dataset_dict.keys():
        if name in file_path:
            return name
    return "Unknown"


def load_file_data(data_index, task_channels):
    num_samples = 0
    data = {}
    outputs = {}
    srs = {}
    durs = {}
    channels = {}
    datasets = {}
    failed_samples = []

    for sample in tqdm(data_index, desc="Loading data", position=0, leave=True):
        try:
            # Load and concatenate dataframe
            input_files = sample["input"]

            df = pd.DataFrame()
            for file in input_files:
                if load_mode != 1:
                    file = prefix_filepath + file
                else:
                    file = file.replace("/itet-stor/kard", "/itet-stor/maxihuber")
                with open(file, "rb") as f:
                    df_new = pd.read_pickle(f)
                    df = pd.concat([df, df_new], axis=0)
                dataset_name = extract_dataset_name(file, dataset_dict)
                datasets[num_samples] = dataset_name

            start = int(sample["start"])
            length = int(sample["length"]) if "length" in sample else int(sample["end"])
            if load_mode != 1:
                df = df.iloc[start:length, :]
                if short_mode:
                    df = df.iloc[: int(len(df) * 0.5), :]
            else:
                df = df.loc[start : start + length, :]

            # Add metadata
            if len(df) <= 1:
                assert False
            sr = int(
                1 / float(float(df[time_col].iloc[1]) - float(df[time_col].iloc[0]))
            )
            if load_mode != 1:
                outputs[num_samples] = (
                    sample["output"] if "output" in sample else sample["label"]
                )
            else:
                if task_name == "EyeNetPosition":
                    outputs[num_samples] = list(sample["output"].values())
                else:
                    outputs[num_samples] = list(sample["output"].values())[0]
            srs[num_samples] = sr
            durs[num_samples] = len(df) / sr
            channels[num_samples] = list(set(df.columns) & task_channels)
            df = df[channels[num_samples]].astype(float)
            signals = torch.tensor(df.to_numpy(), dtype=torch.float32).T
            data[num_samples] = signals
            num_samples += 1

        except Exception as e:
            print(f"Failed to process sample: {sample}. Error: {e}", file=sys.stderr)
            failed_samples.append(sample)

    return data, outputs, srs, durs, channels, datasets


if load_mode == 0:
    print(json_path, file=sys.stderr)
    train_index, test_index = load_index0(json_path)
elif load_mode == 1:
    train_index, val_index, test_index = load_index1(json_path)
else:
    pass

print(f"Full train size: {len(train_index)}", file=sys.stderr)
print(f"Full test size: {len(test_index)}", file=sys.stderr)

truncate = """
if load_mode == 0:
    train_index = train_index
    test_index = test_index
elif load_mode == 1:
    train_index = train_index
    val_index = val_index
    test_index = test_index
else:
    pass
"""


if load_mode == 0:
    train_data, train_outputs, train_sr, train_dur, train_channels, train_datasets = (
        load_file_data(train_index, task_channels)
    )
    test_data, test_outputs, test_sr, test_dur, test_channels, test_datasets = (
        load_file_data(test_index, task_channels)
    )
elif load_mode == 1:
    train_data, train_outputs, train_sr, train_dur, train_channels, train_datasets = (
        load_file_data(train_index, task_channels)
    )
    val_data, val_outputs, val_sr, val_dur, val_channels, val_datasets = load_file_data(
        val_index, task_channels
    )
    test_data, test_outputs, test_sr, test_dur, test_channels, test_datasets = (
        load_file_data(test_index, task_channels)
    )
else:
    pass


# Label Encoder & Class Weights
from sklearn.preprocessing import LabelEncoder

if isinstance(list(train_outputs.values())[0], str):
    all_outputs = list(set(list(train_outputs.values()) + list(test_outputs.values())))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_outputs)

    print(f"Train classes: {set(train_outputs.values())}", file=sys.stderr)
    print(f"Test classes: {set(test_outputs.values())}", file=sys.stderr)

    # Encode the train and test outputs
    encoded_train_outputs = {
        k: label_encoder.transform([v])[0] for k, v in train_outputs.items()
    }
    encoded_test_outputs = {
        k: label_encoder.transform([v])[0] for k, v in test_outputs.items()
    }

    # Create the output counts map
    train_output_counts = defaultdict(int)
    for output in encoded_train_outputs.values():
        train_output_counts[output] += 1

    test_output_counts = defaultdict(int)
    for output in encoded_test_outputs.values():
        test_output_counts[output] += 1

    full_output_counts = train_output_counts.copy()
    for output, count in test_output_counts.items():
        full_output_counts[output] += count

    print("Full Output Counts:", full_output_counts, file=sys.stderr)

    # Calculate class weights
    total_count = sum(full_output_counts.values())
    class_weights = {
        output: total_count / count for output, count in full_output_counts.items()
    }

    # Convert class weights to a tensor
    weight_tensor = torch.tensor(
        [class_weights[i] for i in range(len(class_weights))], dtype=torch.float
    )
else:
    label_encoder = None
    weight_tensor = None

L.seed_everything(42)

sys.path.append("/home/maxihuber/eeg-foundation/src/models/components/Baselines")

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score
from torchmetrics.functional import mean_squared_error as rmse
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint
from collections import defaultdict


class SimpleDataset(Dataset):
    def __init__(self, data, outputs, datasets, task_type, label_encoder=None):
        self.data = data
        self.outputs = outputs
        self.datasets = datasets
        self.task_type = task_type
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signals = self.data[idx]
        output = self.outputs[idx]
        dataset = self.datasets[idx]

        if self.task_type == "Classification" and self.label_encoder is not None:
            output = self.label_encoder.transform([output])[
                0
            ]  # Encode the output label
            output_tensor = torch.tensor(output, dtype=torch.long)
        else:
            output_tensor = torch.tensor([output], dtype=torch.float32)

        return {
            "signals": signals,
            "output": output_tensor,
            "dataset": dataset,
        }


durs = [df.shape[1] for idx, df in train_data.items()] + [
    df.shape[1] for idx, df in test_data.items()
]
n_chns = [df.shape[0] for idx, df in train_data.items()] + [
    df.shape[0] for idx, df in test_data.items()
]
dur_90 = int(np.percentile(durs, 90))
chn_90 = 128  # int(np.percentile(n_chns, 90))


def pad_tensor(tensor, target_height, target_width):
    current_height, current_width = tensor.shape

    # Pad height if necessary
    if current_height < target_height:
        padding_height = target_height - current_height
        padding = torch.zeros((padding_height, current_width), dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding), dim=0)
    else:
        tensor = tensor[:target_height, :]

    # Pad width if necessary
    if current_width < target_width:
        padding_width = target_width - current_width
        padding = torch.zeros((tensor.shape[0], padding_width), dtype=tensor.dtype)
        tensor = torch.cat((tensor, padding), dim=1)
    else:
        tensor = tensor[:, :target_width]

    return tensor


train_data_pad = {
    k: pad_tensor(signals, chn_90, dur_90) for k, signals in train_data.items()
}
test_data_pad = {
    k: pad_tensor(signals, chn_90, dur_90) for k, signals in test_data.items()
}

full_train_dataset = SimpleDataset(
    train_data_pad,
    train_outputs,
    train_datasets,
    task_type=task_type,
    label_encoder=label_encoder,
)
test_dataset = SimpleDataset(
    test_data_pad,
    test_outputs,
    train_datasets,
    task_type=task_type,
    label_encoder=label_encoder,
)

# Define the split ratio
train_ratio, val_ratio = 0.85, 0.15

# Calculate lengths for train and validation sets
total_size = len(full_train_dataset)
train_size = int(train_ratio * total_size)
val_size = total_size - train_size

# Split the dataset
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

import numpy as np
import mne
import torch

os.makedirs(
    f"/itet-stor/maxihuber/net_scratch/finetune_ckpts/{task_name}", exist_ok=True
)


# Function to resample signals
def resample_signals(data, srs, target_sfreq):
    resampled_data = {}
    for idx, signal in tqdm(data.items(), desc="Resampling signals"):
        signal_numpy = signal.numpy().astype(np.float64)  # Convert to float64
        signal_resampled = mne.filter.resample(signal_numpy, up=target_sfreq / srs[idx])
        resampled_data[idx] = torch.tensor(signal_resampled, dtype=torch.float32)
    return resampled_data


# Function to pad or truncate signals to a common length
def pad_or_truncate_signals(data, common_length):
    for idx, signal in tqdm(data.items(), desc="Pad/Truncate signals"):
        signal_length = signal.shape[1]
        if signal_length < common_length:
            pad_width = common_length - signal_length
            signal_padded = np.pad(signal, ((0, 0), (0, pad_width)), mode="constant")
        else:
            signal_padded = signal[:, :common_length]
        data[idx] = torch.tensor(signal_padded.clone().detach(), dtype=torch.float32)
    return data


# Function to create MNE Epochs object from data
def create_epochs(data, outputs, channels, sfreq=1000, is_classification=True):
    events = []
    event_id = {}
    epochs_data = []
    for idx, signal in tqdm(data.items(), desc="Creating epochs"):
        epochs_data.append(signal.numpy())
        if is_classification:
            if outputs[idx] not in event_id:
                event_id[outputs[idx]] = len(event_id) + 1
            events.append([idx, 0, event_id[outputs[idx]]])
        else:
            events.append([idx, 0, 1])  # Dummy event_id for regression
    events = np.array(events, dtype=int)
    info = mne.create_info(
        ch_names=[f"EEG_{i}" for i in range(chn_90)], sfreq=sfreq, ch_types="eeg"
    )
    epochs = mne.EpochsArray(
        np.array(epochs_data),
        info,
        events=events,
        event_id=event_id if is_classification else None,
    )
    return epochs


# Determine the target sampling frequency (e.g., the highest or mean sampling rate)
target_sfreq = int(max(train_sr.values()))  # or use statistics.mean(train_sr.values())

# Resample train and test data
# train_data_resampled = resample_signals(train_data_pad, train_sr, target_sfreq)
# test_data_resampled = resample_signals(test_data_pad, test_sr, target_sfreq)
train_data_resampled = train_data_pad
test_data_resampled = test_data_pad

# Determine the common length for all signals
common_length = min(
    min(signal.shape[1] for signal in train_data_resampled.values()),
    min(signal.shape[1] for signal in test_data_resampled.values()),
)

# Pad or truncate train and test data to the common length
train_data_padded = pad_or_truncate_signals(train_data_resampled, common_length)
test_data_padded = pad_or_truncate_signals(test_data_resampled, common_length)

# Convert train and test data to MNE Epochs
is_classification = isinstance(list(train_outputs.values())[0], str)
epochs_train = create_epochs(
    train_data_padded,
    train_outputs,
    list(train_channels.values()),
    target_sfreq,
    is_classification,
)
epochs_test = create_epochs(
    test_data_padded,
    test_outputs,
    list(test_channels.values()),
    target_sfreq,
    is_classification,
)

# Create an xDAWN instance and fit it to the training data
xdawn = Xdawn(n_components=2, correct_overlap=False, reg=0.1)  # Adding regularization
xdawn.fit(epochs_train)

# Transform the data using xDAWN
X_train_xdawn = xdawn.transform(epochs_train)
X_test_xdawn = xdawn.transform(epochs_test)

# Save xDAWN model parameters
with open(
    f"/itet-stor/maxihuber/net_scratch/finetune_ckpts/{task_name}/xdawn_model.pkl", "wb"
) as f:
    pickle.dump(xdawn, f)

# Flatten the transformed data for LDA input
n_epochs_train, n_components, n_times = X_train_xdawn.shape
X_train_xdawn = X_train_xdawn.reshape(n_epochs_train, n_components * n_times)
n_epochs_test, n_components, n_times = X_test_xdawn.shape
X_test_xdawn = X_test_xdawn.reshape(n_epochs_test, n_components * n_times)

if is_classification:
    # Encode labels if they are strings (for classification tasks)
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(list(train_outputs.values()))
    y_test = label_encoder.transform(list(test_outputs.values()))

    # Create an LDA instance and fit it to the transformed training data
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train_xdawn, y_train)

    # Save LDA model parameters
    with open(
        f"/itet-stor/maxihuber/net_scratch/finetune_ckpts/{task_name}/lda_model.pkl",
        "wb",
    ) as f:
        pickle.dump(lda, f)

    # Predict the labels of the test set
    y_pred = lda.predict(X_test_xdawn)

    # Calculate metrics
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"Balanced Accuracy: {balanced_acc}", file=sys.stderr)
else:
    # For regression tasks
    y_train = np.array(list(train_outputs.values()))
    y_test = np.array(list(test_outputs.values()))

    # Create a linear regression model and fit it to the transformed training data
    lr = LinearRegression()
    lr.fit(X_train_xdawn, y_train)

    # Save Linear Regression model parameters
    with open(
        f"/itet-stor/maxihuber/net_scratch/finetune_ckpts/{task_name}/linear_regression_model.pkl",
        "wb",
    ) as f:
        pickle.dump(lr, f)

    # Predict the values of the test set
    y_pred = lr.predict(X_test_xdawn)

    # Calculate metrics
    rmse_value = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse_value}", file=sys.stderr)

# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    # "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    # "Neural Net (MLP)",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, random_state=42),
    SVC(gamma=2, C=1, random_state=42),
    # GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
    DecisionTreeClassifier(max_depth=5, random_state=42),
    RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1, random_state=42
    ),
    # MLPClassifier(alpha=1, max_iter=100, random_state=42),
    AdaBoostClassifier(algorithm="SAMME", random_state=42),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

X_train = [sample[13] for sample in train_data_pad.values()]
y_train = [output for output in train_outputs.values()]

X_test = [sample[13] for sample in test_data_pad.values()]
y_test = [output for output in test_outputs.values()]

# iterate over classifiers
for name, clf in zip(names, classifiers):
    print(name, file=sys.stderr)
    clf = make_pipeline(StandardScaler(), clf)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test, scoring="balanced_accuracy")
    print(score, file=sys.stderr)
