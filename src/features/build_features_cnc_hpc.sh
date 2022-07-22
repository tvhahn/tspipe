#!/bin/bash
#SBATCH --time=01:40:00 # 10 min
#SBATCH --array=1-80
#SBATCH --cpus-per-task=8
#SBATCH --account=rrg-mechefsk
#SBATCH --mem=16G
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $
## How to use arrays: https://docs.computecanada.ca/wiki/Job_arrays

echo "Starting task $SLURM_ARRAY_TASK_ID"

module load python/3.8

PROJECT_DIR=$1

SCRATCH_DATA_DIR=~/scratch/feat-store/data

source ~/featstore/bin/activate

python $PROJECT_DIR/src/features/build_features.py \
    --path_data_dir $SCRATCH_DATA_DIR \
    --dataset cnc \
    --raw_dir_name data_raw_processed \
    --raw_file_name cnc_raw_54.csv \
    --interim_dir_name cnc_features_comp \
    --processed_dir_name cnc_features_comp \
    --feat_file_name cnc_features_54_comp.csv \
    --feat_dict_name comp \
    --n_chunks 80 \
    --n_cores 8 \
    --chunk_index $SLURM_ARRAY_TASK_ID