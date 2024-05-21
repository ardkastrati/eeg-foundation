import os
from socket import gethostname
import sys
import json
from math import ceil
import sys
import yaml

# from parallel_preprocessor import Preprocessor
from utils import LocalLoader, filter_index


def main(data_config, num_chunks, idx):

    print("Starting process", idx, file=sys.stderr)

    ## TODO implement wandb.log(data_preparation_time) here

    TMPDIR = f"{data_config['runs_dir']}/{os.environ['SLURM_ARRAY_JOB_ID']}/tmp"
    os.makedirs(TMPDIR, exist_ok=True)

    """
    index is a list of filepaths to datapoints.

    the data itself is collected from several root_directories,
    which are stored in the index_paths list.

    the index_lens list holds the number of datapoints collected from each root_directory,
    and the index_sizes holds the summed size of each root_directory.

    in the following, we split the index list into num_chunks parts.

    so far, we split it into equal parts, but this is not optimal from 
    a load-balancing perspective, as each worker afterwards gets the same number of
    filepaths to process, which might greatly vary in size.

    ```{code so far}
    chunk_size = ceil(len(index) / num_chunks)
    index_chunk = index[idx * chunk_size : (idx + 1) * chunk_size]

    print(
        f"Processing [{idx * chunk_size}, {(idx + 1) * chunk_size}) on process = {idx}",
        file=sys.stderr,
    )
    ```

    therefore, i am want to implement a more sophisticated splitting strategy,
    which takes the index_lens and index_sizes into account, to balance the load.

    the goal is that a proportional amount of workers is assigned to each root_directory,
    based on the the index_sizes list.

    then, for each root directory, i want to split the index equally among its workers,
    like it has been done so far for the whole index. 
    """

    local_loader = LocalLoader(
        base_stor_dir=data_config["STORDIR"],
    )

    index, index_lens, index_sizes = filter_index(
        index_paths=data_config["data_dir"],
        path_prefix=data_config["path_prefix"],
        min_duration=data_config["min_duration"],
        max_duration=data_config["max_duration"],
        select_sr=data_config["select_sr"],
        select_ref=data_config["select_ref"],
        discard_datasets=data_config["discard_datasets"],
    )

    # Calculate total size for normalization
    total_size = sum(index_sizes)

    # Calculate number of chunks each root_directory should get based on size
    # E.g. 6 workers for tueg data, 4 workers for pkl data (data is distributed to num_chunks workers in total)
    num_chunks_per_directory = [
        ceil((size / total_size) * num_chunks) for size in index_sizes
    ]
    print(num_chunks_per_directory)

    # Adjust chunk numbers to sum up to the total number of chunks available
    while sum(num_chunks_per_directory) > num_chunks:
        max_index = num_chunks_per_directory.index(max(num_chunks_per_directory))
        num_chunks_per_directory[max_index] -= 1
    print(num_chunks_per_directory)

    # Create global index chunks based on calculated per-directory chunks
    start_idx = 0
    global_index_chunks = []
    for dir_idx, chunks in enumerate(num_chunks_per_directory):
        # Compute the size of a chunk for this directory
        # (e.g. 1'000 files per tueg worker, 20'000 files per pkl worker)
        chunk_size = ceil(index_lens[dir_idx] / max(1, chunks))
        for chunk_idx in range(chunks):
            chunk_start = start_idx + chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, start_idx + index_lens[dir_idx])
            global_index_chunks.append(index[chunk_start:chunk_end])
        start_idx += index_lens[dir_idx]

    # Assign chunk to current worker
    len_index_chunks = [len(index_chunk) for index_chunk in global_index_chunks]
    print("Lengths of index chunks:", len_index_chunks, file=sys.stderr)

    # Print information about global chunk distribution
    processed = 0
    for num_worker in range(num_chunks):
        if num_worker < len(global_index_chunks):
            index_chunk = global_index_chunks[num_worker]
            print(
                f"Worker {num_worker} processes {len(index_chunk)} files from index {processed} to {processed + len(index_chunk)}",
                file=sys.stderr,
            )
        else:
            index_chunk = []
            print(f"Worker {num_worker} has no files to process.", file=sys.stderr)
        processed += len(index_chunk)

    # This if-else here is not the best. I just need it quickly now because the script breaks if we have only 1 worker.
    # TODO: change this to a more elegant solution later
    if num_chunks > 1:
        # Process the index_chunk for this idx...
        index_chunk = global_index_chunks[idx] if idx < len(global_index_chunks) else []
    else:
        index_chunk = index
    # (Add your processing logic here)

    # chunk_path is a path to a json file, which in turn is of the form
    # {0: {"path": path_to/signal0.npy,
    #      "sr:" 250,
    #      ...},
    #  1: {...},
    #  ...}
    # channel_set holds all channel names that were present in the index_chunk on this process
    path_to_signal_chunks_index, channel_set, nr_files = local_loader.load(
        index_chunk=index_chunk,
        thread_id=idx,
    )

    print(
        f"Saved {nr_files} signals at {path_to_signal_chunks_index} on process {idx}.",
        file=sys.stderr,
    )

    # we store this list into a json file in the TMPDIR,
    # so that each process can access it in the setup method afterwards
    with open(
        os.path.join(TMPDIR, f"index_path_{gethostname()}_{idx}.txt"), "w"
    ) as file:
        file.write(path_to_signal_chunks_index)

    # Also store the channels that are present in the data on this process
    # -> we need that for the cls_token_map in the network afterwards
    with open(
        os.path.join(data_config["STORDIR"], f"channel_set_{gethostname()}_{idx}.txt"),
        "w",
    ) as file:
        json.dump(list(channel_set), file)

    print(f"Finished loading on process {idx}.", file=sys.stderr)


if __name__ == "__main__":
    num_chunks = int(sys.argv[1])
    idx = int(sys.argv[2])

    # Load main config file
    main_config_file = "/home/maxihuber/eeg-foundation/configs/experiment/maxim.yaml"
    with open(main_config_file, "r") as file:
        config = yaml.safe_load(file)
        data_config = config["data"]

    main(data_config, num_chunks, idx)
