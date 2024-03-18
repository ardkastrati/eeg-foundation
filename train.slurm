#!/bin/bash

#SBATCH --mail-type=NONE         # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/home/maxihuber/eeg-foundation/runs/sbatch/train_%j.out
#SBATCH --error=/home/maxihuber/eeg-foundation/runs/sbatch/train_%j.err
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --nodelist=tikgpu10
#SBATCH --mem=300G
#SBATCH --time=01:00:00

# Exit on errors
set -o errexit

# Initialize Conda environment
source /itet-stor/maxihuber/net_scratch/conda/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate fastenv

# Navigate to your project directory (if necessary)
cd /home/maxihuber/eeg-foundation/

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Binary or script to execute
srun python /home/maxihuber/eeg-foundation/src/train.py experiment=maxim.yaml

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0