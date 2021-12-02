#!/bin/bash
#SBATCH --time=00:30:00 # 30 min
#SBATCH --array=1-10
#SBATCH --cpus-per-task=4
#SBATCH --account=rrg-mechefsk
#SBATCH --mem=8G
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $
## How to use arrays: https://docs.computecanada.ca/wiki/Job_arrays

echo "Starting task $SLURM_ARRAY_TASK_ID"

module load python/3.8

PROJECT_DIR=$1

SCRATCH_DATA_DIR=~/scratch/feat-store/data

source ~/featstore/bin/activate

python $PROJECT_DIR/src/features/build_features.py --path_data_folder $SCRATCH_DATA_DIR --n_chunks 10 --chunk_index $SLURM_ARRAY_TASK_ID