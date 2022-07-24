#!/bin/bash
#SBATCH --time=00:25:00 # 10 minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

PROJECT_DIR=$1

SCRATCH_DATA_DIR=~/scratch/feat-store/data

module load python/3.8
source ~/featstore/bin/activate

python $PROJECT_DIR/src/models/filter.py \
    -p $PROJECT_DIR \
    --path_data_dir ~/scratch/feat-store/data \
    --path_model_dir ~/scratch/feat-store/models \
    --dataset cnc \
    --processed_dir_name cnc_features_comp \
    --feat_file_name cnc_features_54_comp.csv \
    --final_dir_name final_results_cnc \
    --keep_top_n 100 \
    --save_n_figures 0