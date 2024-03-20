import os
import json
import glob


def compress(epoch_dir, file_path):
    """
    Compresses the trace files located at file_path and store them into `epoch_dir/compressed_nomem`
    """

    filename = os.path.basename(file_path)

    with open(file_path, "r") as file:
        trace_data = json.load(file)

    def keep(event):
        return (
            ("dur" not in event or event["dur"] >= 100)
            and ("ph" not in event or event["ph"] != "i")
            and ("ph" not in event or event["ph"] != "f")
            and ("ph" not in event or event["ph"] != "s")
            and ("ph" not in event or event["ph"] != "e")
        )

    # Filter out events -> compression
    filtered_trace_events = [
        event for event in trace_data["traceEvents"] if keep(event)
    ]

    # Replace the traceEvents with the filtered list
    trace_data["traceEvents"] = filtered_trace_events

    compr_dir = os.path.join(epoch_dir, "compressed_nomem")
    os.makedirs(compr_dir, exist_ok=True)

    compr_filename = f"compressed_{filename}"
    compr_filepath = os.path.join(compr_dir, compr_filename)

    with open(compr_filepath, "w") as file:
        json.dump(trace_data, file, indent=2)

    print(f"Saved compressed file to {compr_filepath}")


def compress_epoch(log_base, log_folder, epoch):
    """
    Takes all files at `log_base/log_folder/f"{epoch}_epoch"/noncompressed` that
     - match the f"{epoch}_*.pt.trace.json" pattern,
     - compresses them, and
     - stores them into `log_base/log_folder/f"{epoch}_epoch"/compressed_nomem`
    """
    epoch_dir = os.path.join(log_base, log_folder, f"{epoch}_epoch")
    noncompr_dir = os.path.join(epoch_dir, "noncompressed")
    pattern = os.path.join(noncompr_dir, f"{epoch}_*.pt.trace.json")
    epoch_trace_files = glob.glob(pattern)
    for file_path in epoch_trace_files:
        compress(epoch_dir, file_path)


def compress_all(log_dir):
    if os.path.exists(log_dir):
        for filename in os.listdir(log_dir):
            if os.path.isfile(os.path.join(log_dir, filename)):
                compress(log_dir, filename)
            else:
                print(f"No file {filename}")
    else:
        print(f"No such directory {log_dir}")


if __name__ == "__main__":
    log_dir_list = [
        "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_19-40",
        "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_20-09",
        "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_20-42",
        "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_21-26",
    ]
    log_dir = "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-19_18-16/0_epoch"
    compress_epoch(log_dir, 0)
    # compress_all(log_dir)
