#!/bin/bash

#SBATCH --mail-type=ALL         # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name=process-subjects
#SBATCH --array=0-9       # Creates 10 jobs
#SBATCH --nodes=1         # All tasks on one node
#SBATCH --mem=10G         # Memory per job
#SBATCH --output=/itet-stor/maxihuber/net_scratch/runs/%A/preloading/%A_%a.out
#SBATCH --error=/itet-stor/maxihuber/net_scratch/runs/%A/preloading/%A_%a.err
#SBATCH --nodelist=tikgpu07

# Create directory for job
mkdir -p /itet-stor/maxihuber/net_scratch/runs/$SLURM_ARRAY_JOB_ID

# Exit on errors
set -o errexit

# Initialize Conda environment
source /itet-stor/maxihuber/net_scratch/conda/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate fastenv

# Navigate to your project directory (if necessary)
cd /home/maxihuber/eeg-foundation/

# Define variables
num_chunks=10  # Must correspond to --array

# Now run the Python script
python /home/maxihuber/eeg-foundation/preloading/preload_chunk.py $num_chunks $SLURM_ARRAY_TASK_ID
