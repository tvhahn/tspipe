#!/bin/bash
#SBATCH --time=00:10:00 # 10 min
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

python $PROJECT_DIR/src/features/build_features_cnc.py \
    --path_data_dir $SCRATCH_DATA_DIR \
    --dataset cnc \
    --raw_dir_name data_raw_processed \
    --raw_file_name cnc_raw_54.csv \
    --processed_dir_name cnc_features \
    --interim_dir_name cnc_features \
    --feat_file_name cnc_features_54_test2.csv \
    --feat_dict_name dummy \
    --n_chunks 10 \
    --n_cores 4 \
    --chunk_index $SLURM_ARRAY_TASK_ID