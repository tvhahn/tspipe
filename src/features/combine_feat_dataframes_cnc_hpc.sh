#!/bin/bash
#SBATCH --time=01:20:00 # 10 minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

module load python/3.8

PROJECT_DIR=$1

SCRATCH_DATA_DIR=~/scratch/feat-store/data

source ~/featstore/bin/activate

python $PROJECT_DIR/src/features/combine_feat_dataframes.py \
    --dataset cnc \
    --path_data_dir $SCRATCH_DATA_DIR \
    --interim_dir_name cnc_features_comp_extra \
    --processed_dir_name cnc_features_comp_extra \
    --feat_file_name cnc_features_54_comp_extra.csv \
    --n_cores 10
