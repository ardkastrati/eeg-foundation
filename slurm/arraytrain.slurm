#!/bin/bash

#SBATCH --mail-type=FAIL,END
#SBATCH --output=/home/maxihuber/eeg-foundation/runs/sbatch/train_%j.out
#SBATCH --error=/home/maxihuber/eeg-foundation/runs/sbatch/train_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:6
#SBATCH --ntasks-per-node=6
#SBATCH --cpus-per-task=$(nproc)
#SBATCH --open-mode=append
#SBATCH --time=05:00:00
##SBATCH --mem=100%

set -o errexit

source /itet-stor/maxihuber/net_scratch/conda/etc/profile.d/conda.sh
conda activate fastenv
cd /home/maxihuber/eeg-foundation/

for node in $(scontrol show hostname $SLURM_JOB_NODELIST); do
    echo "Running on node: $node"
    echo "In directory:    $(pwd)"
    echo "Starting on:     $(date)"
    echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "NUM_NODES: ${SLURM_JOB_NUM_NODES}"
done

# Define the hyperparameters
batch_sizes=(4 8 16 32 64)
num_workers=(0 1 2 4 8 16 32 64 128)

# Loop over the hyperparameters
for batch_size in ${batch_sizes[@]}; do
    for num_worker in ${num_workers[@]}; do
        # If num_worker is greater than batch_size, break out of the inner loop
        if [[ $num_worker -gt $((2 * batch_size)) ]]; then
            break
        fi

        echo num_worker: $num_worker, batch_size: $batch_size

        # # Execute the script with the current hyperparameters
        # # wandb login
        # srun python /home/maxihuber/eeg-foundation/src/train.py experiment=maxim.yaml data.batch_size=$batch_size data.num_workers=$num_worker
    done
done

echo "Finished at:     $(date)"
exit 0