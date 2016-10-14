# Code in Tensorflow for Dark Skies Challenge

## Introduction

This tutorial focuses on [Dark Skies - Classification of Nighttime Images](https://www.crowdai.org/challenges/dark-skies-classification-of-nighttime-images), an interesting challenge on CrowdAI.

The goal of this challenge is to use the manually labeled data-set to develop an image classification algorithm that can correctly identify whether an photo shows stars, cities, or other objects. These photos were taken at night. There are 101,554 images in train set and 52,317 images in test set.
<center>
![Spain in the night](https://github.com/LiberiFatali/darkskies-challenge/blob/master/g3doc/spain_portugalpsp.jpg)
*The Iberian Peninsula at night, showing Spain and Portugal. Madrid is the bright spot just above the center. Credits: NASA*
</center>

This tutorial is structured as a hand-on document. Details about machine learning, neural network or convolution neural network aren't included. 

If you are new to machine learning, I recommend you this excellent course by Andrew Ng from Cousera: https://www.coursera.org/learn/machine-learning. In case you are already familiar with machine learning, the CS231n from Standford (http://cs231n.stanford.edu) will be a great resource when dealing with visual recognition problem. 

This tutorial also uses Convolution Neural Network (CNN), which described extensively in CS231n course. The model is based on Google Inception v3 architecture on Tensorflow framework. Other scripts are written in Python.


## Setup
- First, Tensorflow needs to be installed, then clone the code in the github repo below.

- Instructions to install Tensorflow (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md). You may want to use GPU version for faster processing. For reference, this is my PC configuration when training the model:
```
CPU: Core i7-4790K
RAM: 32GB
GPU: GeForce GTX TITAN X
HDD: 1TB
Ubuntu 14.04, CUDA 8.0, CuDNN 5.1
```


## Code

Github repo contains Tensorflow codes and instructions to train the model on Dark Skies - Classification of Nighttime Images dataset can be found here: https://github.com/LiberiFatali/darkskies-challenge. This repo is based on Google Inception-v3 repo. 

To clone it to local computer, git is required. Then open the Terminal and enter command: 
```
git clone https://github.com/LiberiFatali/darkskies-challenge
```

## Data

List of images can be downloaded from CrowdAI website: https://www.crowdai.org/challenges/dark-skies-classification-of-nighttime-images/dataset_files. After getting `train.csv` and `test_release.csv` and put them in cloned folder, run script `dk_download.py` to download actual images from Nasa.

When train images are downloaded, there should be a 'train' folder with 7 sub-folders that are: `astronaut, aurora, black, city, none, stars, unknown`. This is necessary for later step that convert images to Tensorflow format data. To download test images, just switch to function downloadTestFile and test images will be stored in 'test_release' folder.

## Approach and Model

I use Inception v3 architecture as described in “Rethinking the Inception Architecture for Computer Vision” by Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna. The original paper can be access here: https://arxiv.org/abs/1512.00567.  
<center>
![inception_v3_architecture.png](https://github.com/LiberiFatali/darkskies-challenge/blob/master/g3doc/inception_v3_architecture.png)
*Schematic diagram of Inception V3*
</center>

To speed up the training process, I use the model that is trained on ImageNet 1k dataset. It is released by Google. These are steps to download:
```
# set $DK_ROOT to the folder that is cloned from darkskies-challenge github repo 
# (https://github.com/LiberiFatali/darkskies-challenge)
# for example: 
DK_ROOT=/home/username/darkskies-challenge
# location of where to place the Inception v3 model
DATA_DIR=$DK_ROOT/inception-v3-model
cd ${DATA_DIR}

# download the Inception v3 model
curl -O http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz
# extract
tar xzf inception-v3-2016-03-01.tar.gz

# this will create a directory called inception-v3 which contains the following files.
> ls inception-v3
README.txt
checkpoint
model.ckpt-157585
```

## Preprocessing images

The main script prepares dataset for Tensorflow format is `dk_build_image_data.py`. Briefly, this script takes a structured directory of images and converts it to a sharded TFRecord that can be read by the Inception model.

The directory of training images is created with following structure:
```
  train/astronaut/ISS013-E-88573.JPG
  train/astronaut/ISS020-E-29183.JPG
  ...
  train/aurora/ISS010-E-19304.JPG
  train/aurora/ISS010-E-33666.JPG
  ...
  train/black/ISS006-E-21563.JPG
  train/black/ISS006-E-21565.JPG
  ...
  train/city/ISS006-E-18390.JPG
  train/city/ISS006-E-21390.JPG
  ...
  train/none/ISS006-E-21548.JPG
  train/none/ISS006-E-21601.JPG
  ...
  train/stars/ISS007-E-15075.JPG
  train/stars/ISS013-E-78712.JPG
  ...
  train/unknown/ISS006-E-21633.JPG
  train/unknown/ISS006-E-22850.JPG
  ...
```	
In parent folder `train`, each unique label has its own sub-folder that holds images belong to this label. 

Once the data is arranged in this directory structure, we can run `dk_build_image_data.py` on the data to generate the shardedTFRecord dataset.

Set `DK_ROOT` to the folder that is cloned from darkskies-challenge github repo. To run `dk_build_image_data.py`, enter following in commands in the terminal:
```
## Prepare dataset , this should be modified for your computer
# here I assume that all folders and files are in $DK_ROOT,
# location to where to save the TFRecord data
OUTPUT_DIRECTORY=$DK_ROOT/tf_record
# location of downloaded images
TRAIN_DIR=$DK_ROOT/train
 
# please see below for label file
LABELS_FILE=$DK_ROOT/labels.txt 

# build the preprocessing script.
cd $DK_ROOT
bazel build -c opt inception/dk_build_image_data 

# convert the data. 
bazel-bin/inception/dk_build_image_data --train_directory="${TRAIN_DIR}" --output_directory="${OUTPUT_DIRECTORY}" --labels_file="${LABELS_FILE}" --train_shards=96 --num_threads=8
```

The `$LABELS_FILE` will be a text file that is read by the script that provides a list of all of the labels. Concretely, `$LABELS_FILE` contained the following data:
```
astronaut
aurora
black
city
none
stars
unknown
```
Note that each row of each label corresponds with the entry in the final classifier in the model. That is, the astronaut corresponds to the classifier for entry 1; aurora is entry 2, etc. We skip label 0 as a background class.

After running this script produces files that look like the following:
```
  $TRAIN_DIR/train-00000-of-00096
  $TRAIN_DIR/train-00001-of-00096
  ...
  $TRAIN_DIR/train-00095-of-00096
```
where 96 is the number of shards specified for darkskies-challenge dataset. We aim for selecting the number of shards such that roughly 1024 images reside in each shard. One this data set is built you are ready to train or fine-tune an Inception model on this data set.

Note, be sure to check `num_examples_per_epoch()` in `dk_data.py` to correspond with your number of downloaded images.


## Training

We are now ready to fine-tune a pre-trained Inception-v3 model on the darkskies-challenge data set. This requires two distinct changes to our training procedure:
1. Build the exact same model as previously except we change the number of labels in the final classification layer.
2. Restore all weights from the pre-trained Inception-v3 except for the final classification layer; this will get randomly initialized instead.

We can perform these two operations by specifying two flags: `--pretrained_model_checkpoint_path` and `--fine_tune`. 

The first flag is a string that points to the path of a pre-trained Inception-v3 model. If this flag is specified, it will load the entire model from the checkpoint before the script begins training.

The second flag `--fine_tune` is a boolean that indicates whether the last classification layer should be randomly initialized or restored. You may set this flag to false if you wish to continue training a pre-trained model from a checkpoint. If you set this flag to true, you can train a new classification layer from scratch.

Putting this all together you can retrain a pre-trained Inception-v3 model on the darkskies-challenge data set with the following commands.
```
## Finetune 
# Build the training binary to run on a GPU. If you do not have a GPU, 
# then exclude '--config=cuda' 
cd $DK_ROOT
bazel build -c opt --config=cuda inception/dk_train 

# Directory where to save the checkpoint and events files. 
FINETUNE_DIR=$DK_ROOT/dk-finetune 
# Directory where preprocessed TFRecord files reside. 
DK_DATA_DIR=$DK_ROOT/tf_record
# Path to the downloaded Inception-v3 model. 
MODEL_PATH=$DK_ROOT/inception-v3-model/model.ckpt-157585 
# Run the fine-tuning on the dark-challenge dataset starting from the pre-trained 
# inception-v3 model. 
bazel-bin/inception/dk_train --train_dir="${FINETUNE_DIR}" --data_dir="${DK_DATA_DIR}" --pretrained_model_checkpoint_path="${MODEL_PATH}" --fine_tune=True --initial_learning_rate=0.001 --batch_size=32 --input_queue_memory_factor=8 –num_gpus=1 --num_epochs_per_decay=20 --max_steps=1000000
```

Fine-tuning a model a separate data set requires significantly lowering the initial learning rate. We set the initial learning rate to 0.001.

Now the training is in progress, it constantly outputs to terminal screen:
```
2016-10-13 11:56:22.949164: step 0, loss = 3.11 (1.6 examples/sec; 20.053 sec/batch)
2016-10-13 11:56:48.508299: step 10, loss = 2.55 (46.3 examples/sec; 0.692 sec/batch)
2016-10-13 11:56:55.458712: step 20, loss = 2.49 (42.9 examples/sec; 0.746 sec/batch)
2016-10-13 11:57:02.557317: step 30, loss = 2.43 (45.7 examples/sec; 0.700 sec/batch)
2016-10-13 11:57:09.584892: step 40, loss = 2.39 (45.1 examples/sec; 0.710 sec/batch)
2016-10-13 11:57:16.581422: step 50, loss = 2.20 (45.7 examples/sec; 0.700 sec/batch)
2016-10-13 11:57:23.572435: step 60, loss = 1.51 (46.2 examples/sec; 0.693 sec/batch)
2016-10-13 11:57:30.571183: step 70, loss = 1.97 (45.6 examples/sec; 0.701 sec/batch)
2016-10-13 11:57:37.520570: step 80, loss = 1.80 (46.0 examples/sec; 0.696 sec/batch)
2016-10-13 11:57:44.490582: step 90, loss = 1.62 (45.9 examples/sec; 0.698 sec/batch)
2016-10-13 11:57:51.469971: step 100, loss = 1.51 (46.0 examples/sec; 0.696 sec/batch)
…
```
The loss should be decreased gradually. I trained this model for 45000 steps.


# Testing
To evaluate trained model on CrowAI Dark Skies challenge test set, use `dk_classify.py` script in the repo. Note that you need to modify `MODEL_CHECKPOINT_PATH`, `folder` and `test_csv` to fit your folder structure.
<center>
![dk_sample_test.png](https://github.com/LiberiFatali/darkskies-challenge/blob/master/g3doc/dk_sample_test.png)
*Sample result when running on test set*
</center>

This model gets 85% accuracy in the challenge leader board.



