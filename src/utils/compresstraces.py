import os
import json
import glob


def compress(log_dir, filename):
    """
    Compresses file in `log_dir/filename` and stores it into `log_dir/compressed/compressed_filename`.
    """

    full_filename = os.path.join(log_dir, filename)

    # Load the original trace file
    with open(full_filename, "r") as file:
        trace_data = json.load(file)

    def keep(event):
        return (
            ("dur" not in event or event["dur"] >= 100)
            and ("ph" not in event or event["ph"] != "i")
            and ("ph" not in event or event["ph"] != "f")
            and ("ph" not in event or event["ph"] != "s")
            and ("ph" not in event or event["ph"] != "e")
        )

    # Filter out short events
    filtered_trace_events = [
        event for event in trace_data["traceEvents"] if keep(event)
    ]

    # Replace the traceEvents with the filtered list
    trace_data["traceEvents"] = filtered_trace_events

    # Ensure the directory exists
    filtered_log_dir = os.path.join(log_dir, "compressed_nomem")
    os.makedirs(filtered_log_dir, exist_ok=True)

    filtered_file_name = f"compressed_{filename}"
    filtered_full_filename = os.path.join(filtered_log_dir, filtered_file_name)

    # Save the modified trace data to a new file
    with open(filtered_full_filename, "w") as file:
        json.dump(trace_data, file, indent=2)

    print(f"Saved compressed file to {filtered_full_filename}")


def compress_epoch(log_dir, epoch):
    pattern = os.path.join(log_dir, f"{epoch}_*.pt.trace.json")
    epoch_trace_files = glob.glob(pattern)
    for file_path in epoch_trace_files:
        filename = os.path.basename(file_path)
        compress(log_dir, filename)


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
    log_dir = (
        "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_19-40/"
    )
    compress_epoch(log_dir, 0)
    # compress_all(log_dir)
