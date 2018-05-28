#!/usr/bin/env bash

# Run this from the base folder
# Must have tensorflow configured and activated

# Download and setup the object detection framework
###################################################
# Follow from: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md
source activate hand_detector
# Clone the tensorflow models from git
git clone -b fix/explicit_cast_range git@github.com:peterroelants/models.git

# COCO API installation
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cd ../..
cp -r cocoapi/PythonAPI/pycocotools models/research/

# Setup framework
cd models/research/
# Protobuf Compilation
protoc object_detection/protos/*.proto --python_out=.
# Add Libraries to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:models/research/:models/research/slim

# Testing the Installation
python object_detection/builders/model_builder_test.py
