#!/bin/bash
if [ ! -d ~/scratch/feat-store ]; then
    echo "feat-store folder in scratch does not exist"
    mkdir -p ~/scratch/feat-store
fi

mkdir -p ~/scratch/feat-store/data/raw
cp -r ./data/raw/milling ~/scratch/feat-store/data/raw