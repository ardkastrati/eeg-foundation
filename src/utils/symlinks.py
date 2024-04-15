import os
import glob


def make_sym_dir(sym_base, sym_folder):
    sym_dir = os.path.join(sym_base, sym_folder)
    os.makedirs(sym_dir, exist_ok=True)
    return sym_dir


def make_symlinks(epoch_dir, epoch, sym_dir):
    merge_dir = os.path.join(epoch_dir, f"noncompressed")

    # take all files in merge_dir, create a folder in sym_dir/{epoch}_epoch/filename and symlink the file into it
    for trace_file in os.listdir(merge_dir):
        trace_path = os.path.join(merge_dir, trace_file)
        sym_path = os.path.join(
            sym_dir, f"{epoch}_epoch", f"{trace_file}_dir", trace_file
        )
        os.makedirs(os.path.dirname(sym_path), exist_ok=True)
        os.symlink(trace_path, sym_path)

    # print(f"Created symlink for {merge_dir}")
    # sym_link_path = os.path.join(sym_dir, f"{epoch}_epoch_noncompressed")
    # os.symlink(merge_dir, sym_link_path)

    # merged_files = glob.glob(os.path.join(merge_dir, "*"))

    # for file_path in merged_files:
    #     # Extract the filename from the file_path
    #     file_name = os.path.basename(file_path)

    #     # Path for the symlink in the 'symlinks' directory
    #     symlink_path = os.path.join(sym_dir, file_name)

    #     # Create a symlink for the file in the 'symlinks' directory
    #     # Use os.path.relpath to create a relative path to the target
    #     relative_target_path = os.path.relpath(file_path, sym_dir)
    #     os.symlink(relative_target_path, symlink_path)
    #     print(f"Created symlink for {file_name}")


def symlink_noncompressed(run_base, sym_dir):
    """
    1. Iterate over all subdirectories in run_base, i.e. run_base/{i}_epoch
    2. For each trace (.json file) in run_base/{i}_epoch/noncompressed/*,
    create a symlink for it to sym_dir/{i}_epoch/{node_nr}_noncompressed/trace
    while creating missing target directories on the way
    """
    for epoch_dir in os.listdir(run_base):
        for global_rank, trace_file in enumerate(
            sorted(os.listdir(os.path.join(run_base, epoch_dir, "noncompressed")))
        ):
            trace_path = os.path.join(run_base, epoch_dir, "noncompressed", trace_file)
            sym_path = os.path.join(sym_dir, epoch_dir, trace_file, trace_file)
            os.makedirs(os.path.dirname(sym_path), exist_ok=True)
            os.symlink(trace_path, sym_path)


def linkintogooglecompressed(from_dir, to_dir):
    # Write a method that takes traces in from_dir and symlinks them into to_dir (same structure, just symlinks to the actual folders)
    # e.g. from_dir=/home/maxihuber/eeg-foundation/googlesheetresults/2024-03-16_19-40 contains dirs like 0_epoch, 1_epoch and so on.
    # for each of these, create a directory in to_dir=/home/maxihuber/eeg-foundation/googlesheetresults/symlinks/2024-03-16_19-40/(0_epoch, 1_epoch, ...)
    # and then symlink (ln -s) the files in the original directory to the new directory
    # now, write the code
    for epoch_dir in os.listdir(from_dir):
        sym_dir = os.path.join(to_dir, epoch_dir)
        os.makedirs(sym_dir, exist_ok=True)
        for trace_dir_name in os.listdir(os.path.join(from_dir, epoch_dir)):
            trace_path = os.path.join(from_dir, epoch_dir, trace_dir_name)
            sym_path = os.path.join(sym_dir, trace_dir_name)
            os.symlink(trace_path, sym_path)


if __name__ == "__main__":
    runs = ["2024-03-23_16-17"]

    # move into googlesheetresults
    for run in runs:
        print(f"Processing run {run}")
        profilerbase = "/itet-stor/maxihuber/net_scratch/profiling/profileroutput"
        run_base = os.path.join(profilerbase, run)
        sym_base = "/home/maxihuber/eeg-foundation/googlesheetresults"
        sym_dir = os.path.join(sym_base, run)
        os.makedirs(sym_dir, exist_ok=True)
        symlink_noncompressed(run_base, sym_dir)

    # symlink into googlesheetresults/symlinks (thereby tricking tensorboard)
    from_dir = "/home/maxihuber/eeg-foundation/googlesheetresults"
    to_dir = "/home/maxihuber/eeg-foundation/googlesheetresults/symlinks"
    for run in runs:
        linkintogooglecompressed(os.path.join(from_dir, run), os.path.join(to_dir, run))


# if __name__ == "__main__":
#     print("in main")
#     runs = ["2024-03-16_20-42"]
#     from_dir = "/home/maxihuber/eeg-foundation/googlesheetresults"
#     to_dir = "/home/maxihuber/eeg-foundation/googlesheetresults/symlinks"
#     for run in runs:
#         linkintogooglecompressed(os.path.join(from_dir, run), os.path.join(to_dir, run))

# if __name__ == "__main__":
#     log_dir_list = [
#         "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_19-40",
#         "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_20-09",
#         "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_20-42",
#         "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_21-26",
#     ]
#     merged_dir = "/itet-stor/maxihuber/net_scratch/profiling/profileroutput"
#     run_folder = "2024-03-16_19-40"
#     symlink_basedir = "/home/maxihuber/eeg-foundation/temp"
#     make_symlinks(merged_dir, run_folder, symlink_basedir)
