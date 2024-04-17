import os, glob
import shutil


def move_by_epoch(log_base, log_folder, epoch):
    """
    Create a folder called f"{epoch}_epoch"
    For each file matching this epoch in `log_dir/log_folder`, acc. to this pattern f"{epoch}_{node_id}_{servername}.pt.trace.json", move it to this folder
    """
    noncompr_dir = os.path.join(log_base, log_folder, f"{epoch}_epoch", "noncompressed")
    os.makedirs(noncompr_dir, exist_ok=True)
    epoch_filenames = glob.glob(
        os.path.join(log_base, log_folder, f"{epoch}_*_*.pt.trace.json")
    )
    for filename in epoch_filenames:
        shutil.move(filename, noncompr_dir)


def make_epoch_dir(log_base, log_folder, epoch):
    epoch_dir = os.path.join(log_base, log_folder, f"{epoch}_epoch")
    os.makedirs(epoch_dir, exist_ok=True)
    return epoch_dir


if __name__ == "__main__":
    # log_dir = (
    #     "/itet-stor/maxihuber/net_scratch/profiling/profileroutput/2024-03-19_18-16/"
    # )
    # epochs = [0, 1, 2, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
    # group_by_epoch(log_dir, epochs)
    pass
