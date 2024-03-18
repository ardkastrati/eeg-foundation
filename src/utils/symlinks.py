import os
import glob


def make_symlinks(basedir, run_folder, symlink_basedir):
    merged_dir = os.path.join(basedir, run_folder, "merged0")
    symlinks_dir = os.path.join(symlink_basedir, f"symlinks0_{run_folder}")

    # Create the 'symlinks' directory if it doesn't exist
    if not os.path.exists(symlinks_dir):
        os.makedirs(symlinks_dir)

    merged_files = glob.glob(os.path.join(merged_dir, "*"))

    for file_path in merged_files:
        # Extract the filename from the file_path
        file_name = os.path.basename(file_path)

        # Path for the symlink in the 'symlinks' directory
        symlink_path = os.path.join(symlinks_dir, file_name)

        # Create a symlink for the file in the 'symlinks' directory
        # Use os.path.relpath to create a relative path to the target
        relative_target_path = os.path.relpath(file_path, symlinks_dir)
        os.symlink(relative_target_path, symlink_path)
        print(f"Created symlink for {file_name}")


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
