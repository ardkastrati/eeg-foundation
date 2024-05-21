#!/bin/bash

#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE         # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/maxihuber/net_scratch/runs/%j/train_%j.out
#SBATCH --error=/itet-stor/maxihuber/net_scratch/runs/%j/train_%j.err
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --time=03:00:00
#SBATCH --nodelist=tikgpu10

# Create directories for job
mkdir -p /itet-stor/maxihuber/net_scratch/runs/$SLURM_JOB_ID
mkdir /itet-stor/maxihuber/net_scratch/runs/$SLURM_JOB_ID/metrics

# Exit on errors
set -o errexit

# Initialize Conda environment
source /itet-stor/maxihuber/net_scratch/conda/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate fastenv

# Navigate to your project directory (if necessary)
cd /home/maxihuber/eeg-foundation/

# Send some noteworthy information to the output log
for node in $(scontrol show hostname $SLURM_JOB_NODELIST); do
    echo "Running on node: $node"
    echo "In directory:    $(pwd)"
    echo "Starting on:     $(date)"
    echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
    echo "NUM_NODES: ${SLURM_JOB_NUM_NODES}"
done

# Binary or script to execute
wandb login
srun python /home/maxihuber/eeg-foundation/src/train.py experiment=rope.yaml

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0