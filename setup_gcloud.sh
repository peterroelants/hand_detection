#!/usr/bin/env bash

source activate hand_detector

# Setup Google cloud

# Upload data
gsutil cp input_data/hands_train.record gs://hand-detection/data/hands_train.record
gsutil cp input_data/hands_test.record gs://hand-detection/data/hands_test.record
gsutil cp input_data/label_map.pbtxt gs://hand-detection/data/label_map.pbtxt

# Upload pretrained model
gsutil cp downloaded_models/ssd_inception_v2_coco_2017_11_17/model.ckpt.* gs://hand-detection/data/

# Upload config
gsutil cp config/ssd_inception_v2_hands_gcloud.config \
    gs://hand-detection/data/ssd_inception_v2_hands_gcloud.config

# Package code
cd models/research/
python setup.py sdist
cd slim && python setup.py sdist
