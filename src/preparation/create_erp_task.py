import numpy as np
import json
from tqdm import tqdm
import os
from collections import defaultdict

datasets = ["MI_Limb", "MI_BBCI_IV_Berlin", "MI_BBCI_IV_Graz_a", "MI_BBCI_IV_Graz_b", "MI_II", "MI_LR", "MI_eegmmidb", "MI_HGD", "MI_Two", "MI_SCP", "MI_GVH", "MI_GAL", "MI_ULM", "MI_SCI", "RS_ALPHA", "RS_SPIS"]
# Some statistics
statistics = defaultdict(list) # For each dataset, the average number of labels
statistics2 = defaultdict(lambda: defaultdict(list)) # For each dataset, for each label, the average number of samples

erp_overall = json.load(open("/itet-stor/kard/net_scratch/eeg-foundation/src/preparation/erp_overall.json"))

for sample in erp_overall["test"]:
    for dataset in datasets:
        if dataset in list(sample.keys())[0]:
            properties = sample[list(sample.keys())[0]]
            statistics[dataset].append(len(properties["labels"]))
            if properties["length"] > 0:
                statistics2[dataset][tuple(properties["labels"][i][0] for i in range(len(properties["labels"])))].append(properties["length"])

"""
for dataset in datasets:
    print("Statistics" + str(dataset) +  "="*50)
    print(dataset)
    print(np.mean(statistics[dataset]))
    for key, value in statistics2[dataset].items():
        print(key, np.mean(value))
    print("="*50)

print("Statistics" + "="*50)
"""

def create_task(labels, name):
    # TASK 1: Left hand vs Right hand vs Feet
    mi_task_1 = {"train": [], "test": []}
    for sample in erp_overall["train"]:
        path = list(sample.keys())[0]
        properties = sample[path]
        length_seconds = properties["length"]
        if length_seconds > 25:
            continue
        events = tuple(properties["labels"][i][0] for i in range(len(properties["labels"])) if properties["labels"][i][0] != 'unknown' and properties["labels"][i][0] != 'rest')
        for l in labels: # For our task labels
            if [l in event for event in events].count(True): # If the thing happens in one of the events during the trial
                mi_task_1["train"].append({"input": path, "output": l, "start": 0, "length": -1, "length_seconds": length_seconds})
                break

    for sample in erp_overall["test"]:
        path = list(sample.keys())[0]
        properties = sample[path]
        label = tuple(properties["labels"][i][0] for i in range(len(properties["labels"])) if properties["labels"][i][0] != 'unknown' and properties["labels"][i][0] != 'rest')
        length_seconds = properties["length"]
        # if length seconds is not NaN
        if length_seconds > 25:
            continue
        events = tuple(properties["labels"][i][0] for i in range(len(properties["labels"])) if properties["labels"][i][0] != 'unknown' and properties["labels"][i][0] != 'rest')
        for l in labels: # For our task labels
            print(events)
            if [l in event for event in events].count(True): # If the thing happens in one of the events during the trial
                mi_task_1["test"].append({"input": path, "output": l, "start": 0, "length": -1, "length_seconds": length_seconds})
                break
    json.dump(mi_task_1, open(f"/itet-stor/kard/deepeye_storage/foundation_tasks/erp/{name}.json", "w"))

#Task 1: Body parts
create_task(labels = ["with event-related potential", "without event-related potential"], name = "erp_task_1")
