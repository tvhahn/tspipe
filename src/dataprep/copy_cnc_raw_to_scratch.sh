#!/bin/bash
DIR="~/scratch/feat-store"
if [ ! -d "$DIR" ]; then
    echo "feat-store folder in scratch does not exist"
    mkdir ~/scratch/feat-store/data/raw
fi

cd
cd

cp -r ./data/raw/cnc ~/scratch/feat-store/data/raw