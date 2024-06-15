#!/bin/bash

#SBATCH --mail-type=ALL         # Mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name=tasks
#SBATCH --array=0-15            # Adjust based on the number of function calls
#SBATCH --mem=15G               # Memory per job
#SBATCH --output=/itet-stor/kard/net_scratch/DeepEye/logs/cluster_log/%A/%A_%a.out
#SBATCH --error=/itet-stor/kard/net_scratch/DeepEye/logs/cluster_log/%A/%A_%a.err

# Create a unique directory for this entire job array run using $SLURM_ARRAY_JOB_ID
mkdir -p /itet-stor/kard/net_scratch/DeepEye/logs/cluster_log/$SLURM_ARRAY_JOB_ID

# Define the list of task names
TASKS=(
    "eegeyenet_left_right_task_new"
    "eegeyenet_direction_task_new"
    "eegeyenet_position_task_new"
    "detrtime_segmentation_task_new_zuco"
    "detrtime_segmentation_task_new_Movie"
    "detrtime_segmentation_task_new_Dots"
    "controlled_left_right_task"
    "controlled_saccade_task"
    "controlled_saccade_task_h1"
    "controlled_saccade_task_h2"
    "controlled_fixation_task"
    "controlled_fixation_task_h1"
    "controlled_fixation_task_h2"
    "controlled_segmentation_task"
    "controlled_segmentation_task_h1"
    "controlled_segmentation_task_h2"
)

# Get the task name based on the array index
TASK_NAME=${TASKS[$SLURM_ARRAY_TASK_ID]}

# Debugging output to verify the function call
echo "Running function: $TASK_NAME"

# Run the Python script with the current task name
python create_task.py $TASK_NAME