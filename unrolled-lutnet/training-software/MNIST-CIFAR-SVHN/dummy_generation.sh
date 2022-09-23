#!/bin/bash

#Train = sys.argv[1]
#REG = sys.argv[2]
#Retrain = sys.argv[3]
#LUT = sys.argv[4]
#BINARY = sys.argv[5]
#trainable_means = sys.argv[6]
#Evaluate = sys.argv[7]
#epochs = sys.argv[8]
#
echo -e "Please enter dataset (CIFAR-10/SVHN/MNIST/RAILSEM):"
read dataset
echo -e "Generating dummy_lutnet.h5"

python Binary.py ${dataset} True True False True False True False 1 > output.txt
cp models/${dataset}/2_residuals.h5 models/${dataset}/scripts/dummy_lutnet.h5

#bnn regularities
#True True False False False True False


# ""
# Train = sys.argv[2] == 'True'
# REG = sys.argv[3] == 'True'
# Retrain = sys.argv[4] == 'True'
# LUT = sys.argv[5] == 'True'
# BINARY = sys.argv[6] == 'True'
# trainable_means = sys.argv[7] == 'True'
# Evaluate = sys.argv[8] == 'True
# """