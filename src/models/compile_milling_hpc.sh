#!/bin/bash
#SBATCH --time=00:05:00 # 10 minutes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --account=rrg-mechefsk
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

PROJECT_DIR=$1

SCRATCH_DATA_DIR=~/scratch/feat-store/data

module load python/3.8
source ~/featstore/bin/activate

python $PROJECT_DIR/src/models/compile.py \
    -p $PROJECT_DIR \
    --n_cores 8 \
    --interim_dir_name interim_results_milling \
    --final_dir_name final_results_milling

