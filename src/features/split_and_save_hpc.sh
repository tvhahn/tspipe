#!/bin/bash
#SBATCH --time=00:10:00 # 10 minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

module load python/3.8

PROJECT_DIR=$1

SCRATCH_DATA_DIR=~/scratch/feat-store/data

source ~/featstore/bin/activate

python $PROJECT_DIR/src/features/select_feat_and_scale.py \
    --path_data_folder $SCRATCH_DATA_DIR