#!/bin/bash

#SBATCH --mail-type=ALL                 # Mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=/itet-stor/maxihuber/net_scratch/runs/%A/preloading/%A_%a.out
#SBATCH --error=/itet-stor/maxihuber/net_scratch/runs/%A/preloading/%A_%a.err
#SBATCH --job-name=process-and-train    # Combined job name
#SBATCH --array=0-4                     # Creates 5 jobs for preloading
#SBATCH --nodes=1                       # All tasks on one node for preloading
#SBATCH --mem=50G                       # Memory per array job
#SBATCH --nodelist=tikgpu05             # Node selection for preloading

# Create directory for job
mkdir -p /itet-stor/maxihuber/net_scratch/runs/$SLURM_ARRAY_JOB_ID

# Exit on errors
set -o errexit

# Initialize Conda environment
source /itet-stor/maxihuber/net_scratch/conda/etc/profile.d/conda.sh

# Activate Conda environment
conda activate fastenv

# Navigate to project directory (if necessary)
cd /home/maxihuber/eeg-foundation/

# Define variables
num_chunks=$(($SLURM_ARRAY_TASK_MAX + 1))

# Run the Python script for preloading
python /home/maxihuber/eeg-foundation/src/utils/preloading/preload_chunk.py $num_chunks $SLURM_ARRAY_TASK_ID

# Check if this is the last array job, if so continue to training
if [[ $SLURM_ARRAY_TASK_ID -eq $((num_chunks - 1)) ]]; then
    # Check if all array jobs are complete
    ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID

    echo "Waiting for all array jobs to complete..."
    while true; do
        # Count the number of incomplete array tasks
        incomplete_tasks=$(squeue -h -j ${ARRAY_JOB_ID} | grep -c -E ' R| PD| CG ')

        # All except this task are done, we can proceed with training
        # (this process has already returned from the preload script)
        if [[ ${incomplete_tasks} -eq 1 ]]; then
            break
        fi

        echo "Number of incomplete tasks: ${incomplete_tasks}"
        sleep 10
    done

    # Now proceed with the training job
    echo "All array jobs are complete. Proceeding with training."

    # Create directories for training job
    TRAIN_JOB_DIR="/itet-stor/maxihuber/net_scratch/runs/${ARRAY_JOB_ID}/training"
    mkdir -p ${TRAIN_JOB_DIR}/metrics

    # Load conda environment again for training
    source /itet-stor/maxihuber/net_scratch/conda/etc/profile.d/conda.sh
    conda activate fastenv

    # Set up SLURM training job settings
    export SLURM_JOB_ID=${SLURM_ARRAY_JOB_ID}
    export SLURM_OUTPUT="${TRAIN_JOB_DIR}/train_${ARRAY_JOB_ID}.out"
    export SLURM_ERROR="${TRAIN_JOB_DIR}/train_${ARRAY_JOB_ID}.err"
    export NODES=1
    export GPU=1
    export CPUS=10
    export MEM=100G

    # Training job configuration
    sbatch <<EOF
#!/bin/bash
#SBATCH --output=${SLURM_OUTPUT}
#SBATCH --error=${SLURM_ERROR}
#SBATCH --nodes=${NODES}
#SBATCH --gres=gpu:${GPU}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --open-mode=append
#SBATCH --mem=${MEM}
#SBATCH --nodelist=tikgpu05

# Exit on errors
set -o errexit

# Function to execute upon script exit
cleanup() {
    python /home/maxihuber/eeg-foundation/src/utils/preloading/cleanup.py
    echo "Finished at:     $(date)"
}

trap cleanup EXIT

# Initialize Conda environment
source /itet-stor/maxihuber/net_scratch/conda/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate fastenv

# Navigate to your project directory (if necessary)
cd /home/maxihuber/eeg-foundation/

# Send some noteworthy information to the output log
for node in \$(scontrol show hostname \$SLURM_JOB_NODELIST); do
    echo "Running on node: \$node"
    echo "In directory:    \$(pwd)"
    echo "Starting on:     \$(date)"
    echo "SLURM_JOB_ID:    \${SLURM_JOB_ID}"
    echo "CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES}"
    echo "NUM_NODES: \${SLURM_JOB_NUM_NODES}"
done

# Binary or script to execute
wandb login
srun python /home/maxihuber/eeg-foundation/src/train.py experiment=maxim.yaml || echo "Training script failed with exit code $?"

# End the script with exit code 0
exit 0
EOF

fi
