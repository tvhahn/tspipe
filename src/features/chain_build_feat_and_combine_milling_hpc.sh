#!/bin/bash
PROJECT_DIR=$1

JOBID1=$(sbatch --parsable src/features/build_features_milling_hpc.sh $PROJECT_DIR)
sbatch --dependency=afterok:$JOBID1 src/features/combine_feat_dataframes_milling_hpc.sh $PROJECT_DIR