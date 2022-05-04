#!/bin/bash
PROJECT_DIR=$1

NOW_TIME=$2

SCRATCH_DATA_DIR=~/scratch/feat-store/data

module load python/3.8
source ~/featstore/bin/activate

python $PROJECT_DIR/src/models/train.py --save_dir_name interim_results_$NOW_TIME --path_data_dir $SCRATCH_DATA_DIR --rand_search_iter 4 --feat_selection

