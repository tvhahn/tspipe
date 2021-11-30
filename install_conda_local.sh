#!/bin/bash
conda install -n base -c conda-forge mamba
mamba env create -f envfeatstore.yml
eval "$(conda shell.bash hook)"
conda activate featstore
pip install -e .