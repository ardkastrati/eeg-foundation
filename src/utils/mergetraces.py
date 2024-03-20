import os
import json
import glob


def merge_trace_files(epoch_dir, epoch):
    # Pattern to match all trace files for the specified epoch across all worker IDs
    pattern = os.path.join(
        epoch_dir, "compressed_nomem", f"compressed_{epoch}_*.pt.trace.json"
    )
    trace_files = glob.glob(pattern)

    merged_trace = None

    for trace_file in trace_files:
        with open(trace_file, "r") as f:
            trace_data = json.load(f)

            if merged_trace is None:
                # Use the first file as the base for merging
                merged_trace = trace_data
            else:
                # Merge traceEvents from other files
                merged_trace["traceEvents"].extend(trace_data["traceEvents"])

    merge_dir = os.path.join(epoch_dir, f"{epoch}_epoch_merged")
    os.makedirs(name=merge_dir, exist_ok=True)

    # Save the merged trace to a new file
    if merged_trace:
        output_filename = os.path.join(merge_dir, f"merged_{epoch}.pt.trace.json")
        with open(output_filename, "w") as f:
            json.dump(merged_trace, f, indent=2)
        print(f"Merged trace file saved to {output_filename}")
    else:
        print("No trace files found for the specified epoch.")


if __name__ == "__main__":
    log_dir_list = [
        "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_19-40",
        "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_20-09",
        "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_20-42",
        "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_21-26",
    ]
    log_dir = "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-19_18-16/0_epoch"
    merge_trace_files(log_dir, 0)
    # for epoch in [0, 1, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]:
    #     merge_trace_files(log_dir, epoch)
