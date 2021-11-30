#!/bin/bash
cd
cd scratch/

mkdir -p feat-store/data/raw
cd feat-store/data/raw

# UC Berkeley milling data set
# Check if directory does not exist
if [ ! -d "milling" ]; then
    echo "Downloading milling data set"
    mkdir milling
    cd milling
    wget -4 https://ti.arc.nasa.gov/m/project/prognostic-repository/mill.zip
    cd ..
fi
