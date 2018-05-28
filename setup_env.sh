#!/usr/bin/env bash

# Run this from the base folder
# Must have Anaconda Python installed !


# Setup the Python environment
##############################
ENV_NAME="hand_detector"
# Deactivate the current environment
source deactivate
# Remove hand_detector if this exists already
conda remove --yes --name $ENV_NAME --all

# Create new enviornment
echo "Creating $ENV_NAME"
conda create --yes -n $ENV_NAME python=3.6
source activate $ENV_NAME
echo `which python`
echo `python --version`
echo `which pip`
pip install -r requirements.txt

mkdir -p training_local/
