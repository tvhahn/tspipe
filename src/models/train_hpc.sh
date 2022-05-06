#!/bin/bash
#SBATCH --time=00:5:00 # 30 min
#SBATCH --array=1-20
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

PROJECT_DIR=$1

NOW_TIME=$2

SCRATCH_DATA_DIR=~/scratch/feat-store/data

module load python/3.8
source ~/featstore/bin/activate

python $PROJECT_DIR/src/models/train.py --save_dir_name interim_results_milling --date_time $NOW_TIME --path_data_dir $SCRATCH_DATA_DIR --rand_search_iter 300 --feat_selection

