#!/bin/bash
#SBATCH --time=00:05:00 # 30 min
#SBATCH --array=1-3
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --mail-type=ALL               # Type of email notification- BEGIN,END,F$
#SBATCH --mail-user=18tcvh@queensu.ca   # Email to which notifications will be $

PROJECT_DIR=$1

NOW_TIME=$2

module load python/3.8
source ~/featstore/bin/activate

python $PROJECT_DIR/src/models/train.py --save_dir_name interim_results_$NOW_TIME

