# Tensoflow hand detector

Illustration of tensorflow [object detection framework](https://github.com/tensorflow/models/tree/master/research/object_detection) on hand detection.


## Setup

Setup environment by running
```
./setup_env.sh
```

[Install tensorflow dection framework](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) by running:
```
./install_detector_framework.sh
```

Before running each of the scripts or starting jupyter run the following from the base folder.
```
source activate hand_detector
export PYTHONPATH=$PYTHONPATH:models/research/:models/research/slim
```


## Download and process data

This example will be using the
["EgoHands: A Dataset for Hands in Complex Egocentric Interactions"](http://vision.soic.indiana.edu/projects/egohands/) dataset.

Download the data by running:
```
./download_data.sh
```

Next start jupyter by `jupyter notebook` and go do the `input directory`. Run the following notebooks in order:

1. `input_data/ProcessEgoHands.ipynb` to process the ego_hands dataset.
2. `input_data/ConvertToRecord.ipynb` to convert the data to [TFRecord file format](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details).


## training

Download [pretrained model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) with:
```
./download_model.sh
```

### Run locally

Run the training job locally with:
```
python models/research/object_detection/train.py \
  --logtostderr \
  --pipeline_config_path=config/ssd_inception_v2_hands_local.config \
  --train_dir=training_local/
```

Monitor the training with tensorboard by:
```
tensorboard --logdir=training_local/
```

I didn't run the evaluation because the [COCOAPI](https://github.com/cocodataset/cocoapi) has issues running on OSX.


### Run on GCloud

Setup a project on Google Cloud as described at: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md#setting-up-a-project-on-google-cloud

I named my project `object-dectection`, and the bucket `hand-detection` update these name in all the scrips if you are using different names.

Run the following to copy over all the files to gcloud:
```
./setup_gcloud.sh
```

Start the training job with:
```
gcloud ml-engine jobs submit training object_detection_`date +%s` \
   --runtime-version 1.8 \
   --job-dir=gs://hand-detection/train \
   --packages models/research/dist/object_detection-0.1.tar.gz,models/research/slim/dist/slim-0.1.tar.gz \
   --module-name object_detection.train \
   --region europe-west1 \
   --config config/cloud.yml \
   -- \
   --train_dir=gs://hand-detection/train \
   --pipeline_config_path=gs://hand-detection/data/ssd_inception_v2_hands_gcloud.config
```

You can monitor the training with:
```
gcloud auth application-default login
tensorboard --logdir=gs://hand-detection/
```

After training download the model with:
```
gsutil cp "gs://hand-detection/train/model.ckpt-${CHECKPOINT_NUMBER}.*" training_results/
```

Export the graph by running:
```
python models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path config/ssd_inception_v2_hands_gcloud.config \
    --trained_checkpoint_prefix training_results/model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory exported_graphs
```


## Testing

A notebook to test the results visually on test data is provided in `TestResults.ipynb`. A notebook that runs the model on a webcam stream in the notebook is provided in `WebcamDetector.ipynb`.


## Next steps

### Other datasets:

Other hand detection datasets that could potentially be used:

* [Hand detection using multiple proposals](http://www.robots.ox.ac.uk/~vgg/data/hands/)
* [EgoHands: A Dataset for Hands in Complex Egocentric Interactions](http://vision.soic.indiana.edu/projects/egohands/) (Used in this examples).
* [Large-scale Multiview 3D Hand Pose Dataset](http://www.rovit.ua.es/dataset/mhpdataset/)
* [VIVA hand detection benchmark](http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/)
* [The NUS hand posture datasets ](https://www.ece.nus.edu.sg/stfpage/elepv/NUS-HandSet/)
* [2017 Hands in the Million Challenge](http://icvl.ee.ic.ac.uk/hands17/challenge/)
* [NYU Hand Pose Dataset](https://cims.nyu.edu/~tompson/NYU_Hand_Pose_Dataset.htm)


### Other blogposts on object detection with the tensorflow framework:
* https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
* https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e
* https://ai.googleblog.com/2017/06/supercharge-your-computer-vision-models.html
* https://www.oreilly.com/ideas/object-detection-with-tensorflow
