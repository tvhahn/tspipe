#!/bin/bash
PROJECT_DIR=$1
cd $PROJECT_DIR

mkdir -p $PROJECT_DIR/data/raw

# move to raw data folder
cd $PROJECT_DIR/data/raw

# UC Berkeley milling data set
# Check if directory does not exist
if [ ! -d "milling" ]; then
    echo "Downloading milling data set"
    mkdir milling
    cd milling
    wget -4 https://ti.arc.nasa.gov/m/project/prognostic-repository/mill.zip
    cd ..
fi

