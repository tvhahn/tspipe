#!/bin/bash
DIR=~/scratch/feat-store
if [ ! -d $DIR ]; then
    echo "feat-store folder in scratch does not exist"
    mkdir ~/scratch/feat-store
fi

# copy data folder from the git repo project directory to the scratch folder
cp -r ./data ~/scratch/feat-store

cd $DIR/data/raw

# UC Berkeley milling data set
# Check if directory does not exist
if [ ! -d "milling" ]; then
    echo "Downloading milling data set"
    mkdir milling
    cd milling
    wget -4 https://ti.arc.nasa.gov/m/project/prognostic-repository/mill.zip
    cd ..
fi
