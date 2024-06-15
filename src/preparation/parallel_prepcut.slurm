#!/bin/bash

#SBATCH --mail-type=ALL         # Mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name=prepcut      # Job name
#SBATCH --array=0-5            # Adjust based on the number of datasets minus one
#SBATCH --mem=15G               # Memory per job
#SBATCH --output=/itet-stor/kard/net_scratch/eeg-foundation/logs/cluster_log/%A/%A_%a.out
#SBATCH --error=/itet-stor/kard/net_scratch/eeg-foundation/logs/cluster_log/%A/%A_%a.err

# Define the datasets
DATASET_COLLECTION=("MI_Limb")

# Create a unique directory for this entire job array run using $SLURM_ARRAY_JOB_ID
mkdir -p /itet-stor/kard/net_scratch/DeepEye/logs/cluster_log/$SLURM_ARRAY_JOB_ID

# Calculate the actual number of datasets
NUM_DATASETS=${#DATASET_COLLECTION[@]}
if [ $NUM_DATASETS -lt 50 ]; then
    echo "Number of datasets ($NUM_DATASETS) is less than 50. Consider adjusting --array bounds if possible."
elif [ $NUM_DATASETS -gt 50 ]; then
    echo "Error: Number of datasets ($NUM_DATASETS) exceeds 50. Unable to process all datasets. Adjust the bounds."
    exit 1
fi

# Extract dataset from the collection based on the SLURM_ARRAY_TASK_ID
DATASET=${DATASET_COLLECTION[$SLURM_ARRAY_TASK_ID]}

# Debugging output to verify arguments passed to Python script
echo "Running prepcut_dataset.py with dataset=$DATASET"

# Now run the Python script
python prepcut_dataset.py $DATASET