#!/bin/bash
PROJECT_DIR=$1

JOBID1=$(sbatch --parsable src/features/build_features_hpc.sh $PROJECT_DIR)
sbatch --dependency=afterok:$JOBID1 src/data/make_raw_data_hpc.sh $PROJECT_DIR