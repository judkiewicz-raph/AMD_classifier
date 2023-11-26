# AMD_classifier

This project provides a binary classifier for detecting signs of age-related macular degeneration (AMD) in fundus images.

## Prerequisites

You can install the required dependencies using requirements.txt.

## Code
There are two main scripts for training and testing the classifier:

train.py: This script trains the classifier on a dataset of fundus images.

python train.py data_dir --test-dir test_dir

data_dir is the path to the directory containing the training data.

test_dir (optional) is the path to the directory containing the test data. If you do not specify a test directory, the script will only evaluate the model on validation data.


test.py: This script tests the classifier on a dataset of fundus images.

python test.py --probability-threshold 0.89 --model-trained-path model_path

probability-threshold is the probability threshold used to classify images as positive or negative for AMD. A higher threshold will result in fewer false positives, but more false negatives.

model-trained-path is the path to the trained model file.
