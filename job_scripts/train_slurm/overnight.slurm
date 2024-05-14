#!/bin/bash

#SBATCH --mail-type=ALL         # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/home/maxihuber/eeg-foundation/runs/sbatch/train_%j.out
#SBATCH --error=/home/maxihuber/eeg-foundation/runs/sbatch/train_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --open-mode=append
#SBATCH --mem=100G
#SBATCH --time=23:59:00

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
done

# Binary or script to execute
# make visualize
srun python /home/maxihuber/eeg-foundation/development/index_creation/builder_script.py

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0