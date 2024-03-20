import os
import glob


def make_sym_dir(sym_base, sym_folder):
    sym_dir = os.path.join(sym_base, sym_folder)
    os.makedirs(sym_dir, exist_ok=True)
    return sym_dir


def make_symlinks(epoch_dir, epoch, sym_dir):
    merge_dir = os.path.join(epoch_dir, f"{epoch}_epoch_merged")
    sym_link_path = os.path.join(sym_dir, f"{epoch}_epoch_merged")
    os.symlink(merge_dir, sym_link_path)
    # print(f"Created symlink for {merge_dir}")

    # merged_files = glob.glob(os.path.join(merged_dir, "*"))

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


if __name__ == "__main__":
    log_dir_list = [
        "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_19-40",
        "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_20-09",
        "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_20-42",
        "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-16_21-26",
    ]
    merged_dir = "/itet-stor/maxihuber/net_scratch/profiling/profileroutput"
    run_folder = "2024-03-16_19-40"
    symlink_basedir = "/home/maxihuber/eeg-foundation/temp"
    make_symlinks(merged_dir, run_folder, symlink_basedir)
