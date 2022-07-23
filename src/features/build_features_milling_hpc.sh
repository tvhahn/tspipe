#!/bin/bash
#SBATCH --time=00:10:00 # 10 min
#SBATCH --array=1-20
#SBATCH --cpus-per-task=8
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

python $PROJECT_DIR/src/features/build_features.py \
    --path_data_dir $SCRATCH_DATA_DIR \
    --dataset milling \
    --raw_dir_name stride64_len64 \
    --raw_file_name milling.csv.gz \
    --interim_dir_name milling_features_comp \
    --processed_dir_name milling_features_comp \
    --feat_file_name milling_features_comp.csv \
    --feat_dict_name comp \
    --n_chunks 20 \
    --n_cores 8 \
    --chunk_index $SLURM_ARRAY_TASK_ID