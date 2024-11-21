# MNIST Model Training and Testing

![Build Status](https://github.com/pruthvibhoomi/CodeWithAI/actions/workflows/test_model.yml/badge.svg)

This repository contains a simple implementation of a convolutional neural network (CNN) to classify handwritten digits from the MNIST dataset. The model is trained and evaluated using PyTorch.

## Model Overview

The model consists of the following components:

- **Convolutional Layers**: The model uses three convolutional layers to extract features from the input images.
- **Dropout Layers**: Dropout is applied to prevent overfitting during training.
- **Fully Connected Layers**: The output from the convolutional layers is flattened and passed through two fully connected layers to produce the final output.

## Dataset

The MNIST dataset is used for training and testing the model. It consists of 60,000 training images and 10,000 test images of handwritten digits (0-9).

Below are some of the experiment results.
# Added only random rotations to see if it can increase the accuracy, but accuracy reduced to 91%
# Rotations+RandomAffine accuracy reduced to 80%
# Adam + above 2 data augs , accuracy slightly improved to 85%
# Adam + Only RandomRotations , accuracy is 89%
# Without normalization , accuracy slightly reduces to 93.6%

## Experiments with batch_size
# batch_size 64 , accuracy 92%, 94%
# batch_size 70 , 91%
# batch size 45 , 95%
# with NO shuffle , batch_size = 64, acc=93
# with NO suffle , batch_size=50,acc=91; batch_size=40,acc=94 , 
# batch_size=45,acc=95, 47-->94, 46-->92 ,

## Optimizer experimentations

# with lr=0.1 , accuracy decreased to 91%
# with Adam , accuracy was around 93%
# With SGD, gelu , batch_size=23, acc=90% with lr=0.01
# With SGD,gelu,batch_size=23, acc=96 wth lr=0.1

