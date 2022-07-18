#!/bin/bash
PROJECT_DIR=$1

JOBID1=$(sbatch --parsable src/features/build_features_hpc.sh $PROJECT_DIR)
sbatch --dependency=afterok:$JOBID1 src/features/combine_feat_dataframes_hpc.sh $PROJECT_DIR