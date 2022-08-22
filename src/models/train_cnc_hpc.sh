#!/bin/bash
#SBATCH --time=16:10:00 # 30 min
#SBATCH --array=1-95
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

PROJECT_DIR=$1

NOW_TIME=$2

SCRATCH_DATA_DIR=~/scratch/feat-store/data

module load python/3.8
source ~/featstore/bin/activate

python $PROJECT_DIR/src/models/train.py \
    --path_data_dir $SCRATCH_DATA_DIR \
    --dataset cnc \
    --save_dir_name interim_results_cnc \
    --processed_dir_name cnc_features_comp_extra \
    --feat_file_name cnc_features_54_comp_extra.csv \
    --rand_search_iter 5000



