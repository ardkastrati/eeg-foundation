import numpy as np
import json
from tqdm import tqdm
import os
from collections import defaultdict

datasets = ["MI_Limb", "MI_BBCI_IV_Berlin", "MI_BBCI_IV_Graz_a", "MI_BBCI_IV_Graz_b", "MI_II", "MI_LR", "MI_eegmmidb", "MI_HGD", "MI_Two", "MI_SCP", "MI_GVH", "MI_GAL", "MI_ULM", "MI_SCI", "RS_ALPHA", "RS_SPIS"]
# Some statistics

statistics = defaultdict(list) # For each dataset, the average number of labels
statistics2 = defaultdict(lambda: defaultdict(list)) # For each dataset, for each label, the average number of samples

mi_overall = json.load(open("/itet-stor/kard/net_scratch/eeg-foundation/src/preparation/mi_all.json"))
"""
category = "hand"
for sample in mi_overall["train"]:
    for dataset in datasets:
        if dataset in sample["input"][0] and category in sample["label"]:
            statistics2[dataset][sample["label"]].append(sample["length_seconds"])
for sample in mi_overall["test"]:
    for dataset in datasets:
        if dataset in sample["input"][0] and category in sample["label"]:
            statistics2[dataset][sample["label"]].append(sample["length_seconds"])


for dataset in datasets:
    print("Statistics" + str(dataset) +  "="*50)
    print(dataset)
    for key, value in statistics2[dataset].items():
        print(key, np.nanmean(value))
    print("="*50)

print("Statistics" + "="*50)
"""
def create_task(labels, name, limit = 10000):
    mi_task_1 = {"train": [], "test": []}
    for sample in mi_overall["train"]:
        path = sample["input"][0]
        length_seconds = sample["length_seconds"]
        event = sample["label"]
        if length_seconds > limit or length_seconds == np.nan:
            continue
        for l in labels: # For our task labels
            if l in event: # If the thing happens in the event
                mi_task_1["train"].append({"input": [path], "output": l, "start": 0, "length": -1, "length_seconds": length_seconds})
                break
    for sample in mi_overall["test"]:
        path = sample["input"][0]
        length_seconds = sample["length_seconds"]
        event = sample["label"]
        if length_seconds > limit or length_seconds == np.nan:
            continue
        for l in labels: # For our task labels
            if l in event: # If the thing happens in the event
                mi_task_1["test"].append({"input": [path], "output": l, "start": 0, "length": -1, "length_seconds": length_seconds})
                break
    mi_task_1["num_classes"] = len(labels)
    json.dump(mi_task_1, open(f"/itet-stor/kard/deepeye_storage/foundation_tasks/mi/{name}.json", "w"))

#Task 1: Body parts
create_task(labels = ["left hand imagined movement", "right hand imagined movement", "foot imagined movement", "tongue imagined movement", "rest"], name = "mi_task_imagined_body_parts")

#Task 1b: Body parts
create_task(labels = ["left hand movement", "right hand movement", "foot movement", "rest"], name = "mi_task_body_parts")

# Task 2: Left hand vs Right hand
labels = ["left hand movement", "right hand movement"]
create_task(labels, name = "lr_real")

# Task 3: Left hand vs Right hand imaginary
labels = ["left hand imagined movement", "right hand imagined movement"]
create_task(labels, name = "lr_imaginary")

# Task 4: 
labels = ['thumb imagined movement', 'index finger imagined movement', 'middle finger imagined movement', 'ring finger imagined movement', 'pinkie finger imagined movement', 'relax']
create_task(labels, name = "fingers_imaginary")

# Task 5:
labels = ['hand movement elbow flexion', 'hand movement elbow extension']
create_task(labels, name = "flexion_extension_real")

# Task 6
labels = ['hand movement imagined elbow flexion', 'hand movement imagined elbow extension']
create_task(labels, name = "flexion_extension_imaginary")

# Task 7
labels = ['movement palmar grasp', 'movement lateral grasp']
create_task(labels, name = "grasp_real")

# Task 7b
labels = ['imagined palmar grasp', 'imagined lateral grasp']
create_task(labels, name = "grasp_imaginary")

# Task 8
labels = ["movement pronation", "movement supination"]
create_task(labels, name = "pronation_supination_real")

# Task 8
labels = ["imagined pronation", "imagined supination"]
create_task(labels, name = "pronation_supination_imaginary")

# Task 8
labels = ["eye open", "eye closed"]
create_task(labels, name = "eye_open_closed")

# Task 9
labels = ["vertical", "horizontal"]
create_task(labels, name = "eye_vh")

# Task 10 eegeyenet tasks
# They are done


