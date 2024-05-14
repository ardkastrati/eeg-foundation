# EEG Project

## Description
This document should serve as a brief outline of how the project works and is organized.

## Table of Contents
- [Installation / Project Setup](#Installation)
- [Usage](#Usage)

## Installation

### Installing packages

The necessary dependencies can be found in the `fastenv.yml` file.

Run 

```
conda config --set channel_priority flexible
conda env create -f fastenv.yml
```

to install the environment on your machine (maybe adjust the last line in `fastenv.yml` to set the install location).

Run `conda config --set channel_priority strict` to undo the change to `channel_priority`.

Note: You might run into an error when installing the current dev-build (2.3.0.dev) of Lightning (the most recent version is necessary though). Install it manually using `pip install https://github.com/Lightning-AI/lightning/archive/refs/heads/master.zip -U`.

### Adjusting paths
I would quickly search the whole project (Shift + CMD + F) for occurences of `maxihuber` or `schepasc` and adjust it for your personal username.

### timm patch

Look at the brief explanation at the top of the `TEMPLATE_README.md` file for guidance. Essentially, you need to run the `src/timm_patch/timm_patch.sh` script and adjust some paths beforehand.

## Usage

The important parts of the project structure are (more can be found in the subsections below):
- configs/
- src/: train script and subdirectories
- src/data: LightningDataModule implementations (not used anymore), supporting methods for data loading and transformations (e.g. spectrogram saving)
- src/models: LightningModule (in `mae_module.py`) and nn.Model (in `mae_original.py`)
- src/utils: numerous helper methods and scripts

Supporting folders are:
- slurm/: scripts to submit with `sbatch`
- symlinks/: for storing traces which can then be visualized with tensorboard (see below).

### Running training

I have lots of predefined commands in the Makefile.

The most important are:

`make fulltrain`: loads data into node-local memory (or scratch disk) with multiple worker processes (using a SLURM array job), and afterwards automatically starts a normal SLURM sbatch job that runs the training, after which the data is erased from memory again. Everything is automated, and the training script is modified to make use of the array job as well.

Note: the following parameters need only be set in the slurm script and not in the configs or train scripts. I have taken care of that double declaration issue by pasting the slurm environment variables in the `train.py` script instead of manually inserting them. Also, please note that `--gres=gpu` needs to be equal to `--ntasks-per-node`.

```
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
```

`make time`: Returns all running jobs of the user `maxihuber` (adjust for yourself)

`make free`: Prints an overview of the GPUs in the Arton cluster (runs `smon` under the hood)

`make visualize`: Runs the visualization pipeline for trace files of the PyTorch Profiler. You need to put the runs you want ready to be displayed with tensorboard into the `src/utils/visualization_pipeline.py` file (see below).

Outdated:

`make sbatch`: submits the `slurm/train.slurm` script to the SLURM controller. The script then initiates the training. Set it up as you like.

`make train`: Runs the train script in an interactive session. Make sure to connect to a compute node first (e.g. with `make gpuserver`). Using this command only makes sense when working with a single GPU. The SLURM controlled should be used when working with >1 GPUs or longer trainings.


### Configs

The main config file is `configs/experiment/maxim.yaml`. I passed it to my `srun` commands and it is also used in the slurm script submitted by the `make sbatch` command.

It contains the default settings I have been working with and also links some other config files into it.

The most important other config files (I have worked with) are:
- configs/data/mae_test.yaml
- configs/model/mae_small.yaml
- configs/logger/wandb.yaml
- configs/debug/profiler_test.yaml

### Training output

Console output is saved to `configs/experiment/maxim.yaml -> data.runs_dir`. Currently set to `/itet-stor/maxihuber/net_scratch/runs`.

Traces from the pytorch profiler are saved to `configs/debug/profiler_test.yaml -> profile_dir`. Currently set to `/itet-stor/maxihuber/net_scratch/profiling/profileroutput/`.

For both, a directory is created for the current run (i.e. `runs_dir/SLURM_JOB_ID/`) and then the files go in there.

### Wandb

Currently, wandb is setup for `maximhuber` but you should be able to change it to your own account handle. Call `wandb login` (terminal command). It is also automatically called in the slurm script.

### PyTorch Profiler & Tensorboard

I have implemented the profiling of the code in the `src/utils/training_callbacks.py` file. As the name suggests, the profiling is handled through custom callback code.

- If you want to turn the profiler off, you need to `return False` in the `ProfilerCallback.profileTest` method.
- If you want to turn the profiler on, you can customize the situations in which the same method gives `return True`.

For each epoch (in which the method returns `True`), the profiler is started in the `on_train_epoch_start` method and stopped in the `on_train_epoch_end` method, and it is stepped in the `on_train_batch_end` method.

The traces are automatically saved to the directory specified in `configs/debug/profiler_test.yaml -> profile_dir` for each run. As already said, a new subdirectory is created for each run.

If you want to have a look at the generated traces, you need to go to the `src/utils/visualization_pipeline.py` file and paste the `SLURM_JOB_ID` into the `log_folders` list. Then, just run `make visualize`. The traces will be symbolically linked into the `symlinks/` folder. 

Then, start a new tensorboard session through the VS Code command palette (Shift + CMD + P): `Python: Launch Tensorboard`.

When prompted, click "Select another folder" and paste the absolute path to a folder containing the symbolic links to traces of a certain epoch, e.g. `/home/maxihuber/eeg-foundation/symlinks/symlinks_884871/0_epoch` to load the traces of the first epoch.

You can't load all epochs at once as your local machine will most likely run out of memory.

Make sure not to open too many tensorboard sessions, and periodically run `htop` to kill the running tensorboard processes once you're finished. You can filter by "tensorboard" and then kill all associated processes. I usually had to kill the first $n$ where $n = \text{Number of tensorboard sessions started}$. Just closing the tensorboard tabs is not enough.

I am still figuring out how to run tensorboard on a compute node so this `htop` process killing is not necessary anymore.


### Other

Generally, there are a lot of files hanging around from Pascal's side. Some of them are important for finetuning, some are not (especially configs). I have not removed them as they might be useful in the future.
