# For all epochs {i} do:

# 1. Move the trace files inside the 2024-03-19_19-57 folder into
# 2024-03-19_19-57 > {i}_epoch > noncompressed
from movetraces import make_epoch_dir, move_by_epoch

# 2. Compress the trace files inside the {i}_epoch/noncompressed folder into
# 2024-03-19_19-57 > {i}_epoch > compressed_nomem
from compresstraces import compress_epoch

# 3. Collect and merge the compressed trace files inside {i}_epoch/compressed_nomem into
# 2024-03-19_19-57 > {i}_epoch > {i}_epoch_merged
from mergetraces import merge_trace_files

# 4. "ln -s" the folders {i}_epoch_merged into eeg-foundation
from symlinks import make_sym_dir, make_symlinks

log_base = "/itet-stor/maxihuber/net_scratch/profiling/profileroutput"
sym_base = "/home/maxihuber/eeg-foundation/temp"

log_folders = ["2024-03-23_15-19", "2024-03-23_15-36", "2024-03-23_16-02"]
epochs = [0, 1, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30]

for log_folder in log_folders:
    for epoch in epochs:
        epoch_dir = make_epoch_dir(
            log_base=log_base, log_folder=log_folder, epoch=epoch
        )
        move_by_epoch(log_base=log_base, log_folder=log_folder, epoch=epoch)

        compress_epoch(log_base=log_base, log_folder=log_folder, epoch=epoch)

        merge_trace_files(epoch_dir=epoch_dir, epoch=epoch)

        sym_folder = f"symlinks_{log_folder}"
        sym_dir = make_sym_dir(sym_base=sym_base, sym_folder=sym_folder)
        make_symlinks(epoch_dir=epoch_dir, epoch=epoch, sym_dir=sym_dir)

        print(f"Finished the pipeline for epoch {epoch}")
