#!/bin/bash
#SBATCH --time=00:15:00 # 10 minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

module load python/3.8

PROJECT_DIR=$1

SCRATCH_DATA_DIR=~/scratch/feat-store/data

source ~/featstore/bin/activate

python $PROJECT_DIR/src/dataprep/make_dataset_cnc.py \
    -p $PROJECT_DIR \
    --path_data_dir $SCRATCH_DATA_DIR \
    --split_dir_name data_splits \
    --raw_dir_name data_raw_processed \
    --n_cores 16 \
    --tool_no 54