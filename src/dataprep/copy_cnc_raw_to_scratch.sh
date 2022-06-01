#!/bin/bash
if [ ! -d ~/scratch/feat-store ]; then
    echo "feat-store folder in scratch does not exist"
    mkdir -p ~/scratch/feat-store
fi

mkdir -p ~/scratch/feat-store/data/raw
cp ./data/raw/cnc ~/scratch/feat-store/data/raw/cnc

# extract splits.zip
# unzip ~/scratch/feat-store/data/raw/cnc/splits.zip -d ~/scratch/feat-store/data/raw/cnc