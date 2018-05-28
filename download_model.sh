#!/usr/bin/env bash

# Download pretrained model
cd downloaded_models/
wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
tar -xvf ssd_inception_v2_coco_2017_11_17.tar.gz
