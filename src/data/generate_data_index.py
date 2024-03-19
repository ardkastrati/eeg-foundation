import csv, os
import mne
import json


def generate_lookup_file(data_dir, lookup_name, exclude=[]):

    prefix = "/itet-stor/maxihuber/deepeye_storage/foundation/tueg/edf"
    mne.set_log_level("WARNING")
    counter = 0
    edf_paths = []
    lookup = []

    for root, dirs, files in os.walk(top=data_dir):
        for name in files:

            if name.endswith(".edf"):
                p = os.path.join(root, name)

                edf_paths.append(p)

    print("done getting the paths")

    size = len(edf_paths)
    for path in edf_paths:

        data = {}

        info = mne.io.read_raw_edf(path, preload=False)
        channels = []
        for channel_name in info.info["ch_names"]:
            if "EEG" in channel_name:
                channels.append(channel_name)

        sr = info.info["sfreq"]

        counter += 1
        if counter % 10000 == 0:
            print(size - counter)
            print(path)

        ref = ""
        if "01_tcp_ar" in path:
            ref = "AR"
        if "02_tcp_le" in path:
            ref = "LE"
        if "03_tcp_ar_a" in path:
            ref = "ARA"
        if "04_tcp_le_a" in path:
            ref = "LEA"
        shortpath = path.replace(prefix, "")

        duration = info.times[-1]

        data["path"] = shortpath
        data["channels"] = channels
        data["ref"] = ref
        data["sr"] = sr
        data["duration"] = duration

        lookup.append(data)

    with open(
        lookup_name,
        "w",
    ) as file:

        json.dump(lookup, file)


def generate_channel_index(data_path, stor_path):

    all_channels = []
    with open(data_path, "r") as file:

        data = json.load(file)

        for path, channels, ref in data:
            for c in channels:
                if c not in all_channels:
                    all_channels.append(c)
    print(len(all_channels))
    with open(stor_path, "w") as stor_file:
        json.dump(all_channels, stor_file)


def exclude(data_path, to_exclude_path, new_data_path):

    with open(data_path, "r") as file:
        og_data = json.load(file)

    with open(to_exclude_path, "r") as exclude_file:
        exclude_data = json.load(exclude_file)

    new_data = []

    for edf_file in og_data:

        path = edf_file["path"]
        slash_index = path.rfind("/")
        print(slash_index)
        path = path[slash_index + 1 :]
        print(path)
        if path not in exclude_data:
            new_data.append(edf_file)

    with open(new_data_path, "w") as file:
        json.dump(new_data, file)

    print("removed" + str(len(og_data) - len(new_data)))


def gen_first_run_set(data_path, new_data_path):

    with open(data_path, "r") as file:
        og_data = json.load(file)

    new_data = og_data[:150]
    i = 0
    for d in new_data:
        chn = len(d["channels"])
        i = i + chn
    print(i)

    with open(new_data_path, "w") as file:
        og_data = json.dump(new_data, file)


def gen_paths(input_dir="", output_dir=""):

    with open("tueg_json", "r") as file:
        tueg = json.load(file)
    with open(input_dir, "r") as file:
        in_files = json.load(file)
    output = []
    i = 0
    for file in in_files:

        for line in tueg:
            relativ_path = line["path"]
            if file in relativ_path:
                output.append(line)

    print(len(output))
    with open(output_dir, "w") as file:
        json.dump(output, file)


if __name__ == "__main__":
    # generate_lookup_file("/itet-stor/maxihuber/deepeye_storage/foundation/tueg/edf/000", "debug_json")
    # generate_lookup_file("/itet-stor/maxihuber/deepeye_storage/foundation/tueg/edf", "tueg_json")
    # generate_channel_index("/home/maxihuber/eeg-foundation/src/data/tueg_json", "channel_json")
    # generate_lookup_file("/itet-stor/maxihuber/deepeye_storage/foundation/tueg/edf", "tueg_json")
    # exclude("tueg_json", "tuab_paths", "train_without_tuab")
    # gen_first_run_set("train_without_tuab_json", "disc_test_json")
    gen_paths("/home/schepasc/tuab_eval_normal_paths", "tuab_eval_normal_paths_full")
    gen_paths(
        "/home/schepasc/tuab_eval_abnormal_paths", "tuab_eval_abnormal_paths_full"
    )
