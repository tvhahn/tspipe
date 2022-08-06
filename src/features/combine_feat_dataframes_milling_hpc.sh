#!/bin/bash
#SBATCH --time=00:40:00 # 10 minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

module load python/3.8

PROJECT_DIR=$1

SCRATCH_DATA_DIR=~/scratch/feat-store/data

source ~/featstore/bin/activate

python $PROJECT_DIR/src/features/combine_feat_dataframes.py \
    --dataset milling \
    --path_data_dir $SCRATCH_DATA_DIR \
    --interim_dir_name milling_features_comp_stride64_len1024 \
    --processed_dir_name milling_features_comp_stride64_len1024 \
    --feat_file_name milling_features_comp_stride64_len1024.csv \
    --n_cores 10
