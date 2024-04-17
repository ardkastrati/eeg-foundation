import os
import csv


def filter_and_write(in_folder, in_filename, out_folder, run):
    """
    Scrapes the output files for performance metrics so that it needn't be done manually.

    Filter lines from an input file and write the filtered lines to output files.

    Args:
        in_folder (str): The path to the input folder.
        in_filename (str): The name of the input file (without extension).
        out_folder (str): The path to the output folder.
        run (int): The run number.

    Raises:
        Exception: If an error occurs while reading or writing the files.

    Returns:
        None
    """

    # Constructing the full paths for input and output files
    input_path = f"{in_folder}/{in_filename}.out"

    try:
        # Initialize the lists to store the performance metrics
        epoch_train_times = [[] for _ in range(0, 8)]
        epoch_train_throughputs = [[] for _ in range(0, 8)]

        # Read the input file
        with open(input_path, "r") as infile:
            lines = infile.readlines()

        # Extract the node number, epochs, and performance metrics
        for i, line in enumerate(lines):
            if (
                "Ending train epoch" in line and not "Ending profiling" in lines[i + 1]
            ) or "Ending profiling" in line:
                next_three_lines = lines[i + 1 : i + 4]
                # print(next_three_lines)

                node = int(next_three_lines[2].split("on node ")[1].split(" ")[0])
                epoch = int(next_three_lines[2].split("epoch ")[1].split(" ")[-1])

                epoch_train_time = float(
                    next_three_lines[0].split("epoch_train_time:")[1].strip()
                )
                throughput = float(
                    next_three_lines[2].split("throughput=")[1].split(" ")[0].strip()
                )
                # print(node, epoch, epoch_train_time, throughput)

                epoch_train_times[node].append([epoch, epoch_train_time])
                epoch_train_throughputs[node].append((epoch, throughput))

        print(epoch_train_times[0])
        # Write the filtered lines to the output files
        os.makedirs(f"{out_folder}/{run}", exist_ok=True)

        for node, epoch_train_time_list in enumerate(epoch_train_times):
            with open(
                f"{out_folder}/{run}/epoch_train_times_node{node}.csv",
                "w",
                newline="",
            ) as outfile:
                writer = csv.writer(outfile)
                writer.writerow(["epoch", "train_time"])
                writer.writerows(epoch_train_time_list)

        for node, throughput_list in enumerate(epoch_train_throughputs):
            with open(
                f"{out_folder}/{run}/epoch_train_throughputs_node{node}.csv",
                "w",
                newline="",
            ) as outfile:
                writer = csv.writer(outfile)
                writer.writerow(["epoch", "throughput"])
                writer.writerows(throughput_list)

        print(f"Filtered lines have been written for run {run}")
    except Exception as e:
        print(f"An error occurred: {e}")


in_folder = "/itet-stor/maxihuber/net_scratch/runs/_previous"
runs = [855497, 855500, 855506, 855509]
out_folder = "/home/maxihuber/eeg-foundation/temp"

if __name__ == "__main__":
    for run in runs:
        in_filename = f"train_{run}"
        filter_and_write(in_folder, in_filename, out_folder, run)


# def temp():
#     next_three_lines = [
#         "epoch_train_time: 52.19335699081421",
#         "num_samples 155",
#         "throughput=2.9697265885250355 on node 7 on tikgpu10 in epoch 30",
#     ]
#     # print(next_three_lines[2].split("on node "))
#     # print(next_three_lines[2].split("on node ")[0])
#     # print(next_three_lines[2].split("on node ")[0].split())
#     node = int(next_three_lines[2].split("on node ")[1].split(" ")[0])
#     epoch = int(next_three_lines[2].split("epoch ")[1].split(" ")[-1])
#     print(node)
#     print(epoch)

#     epoch_train_time = float(next_three_lines[0].split("epoch_train_time:")[1].strip())
#     throughput = float(
#         next_three_lines[2].split("throughput=")[1].split(" ")[0].strip()
#     )

#     print((node, epoch_train_time))
#     print((node, throughput))


# temp()
