#!/bin/bash

#SBATCH --mail-type=ALL         # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name=process-subjects
#SBATCH --array=0-0%1  # Creates 50 jobs with at most 50 running at a time
#SBATCH --mem=10G         # Memory per job
#SBATCH --output=/itet-stor/maxihuber/net_scratch/runs/%A/%A_%a.out
#SBATCH --error=/itet-stor/maxihuber/net_scratch/runs/%A/%A_%a.err

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
index_path=/itet-stor/maxihuber/net_scratch/stuff/cleaning/index_new_6.json
num_chunks=50  # Must correspond to --array

# Now run the Python script
#python preprocessing/process_chunk.py $index_path $SLURM_ARRAY_TASK_ID $num_chunks
python preprocessing/process_chunk.py $index_path 2 50
