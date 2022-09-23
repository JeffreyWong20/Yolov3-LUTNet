#!/bin/bash

echo -e "Please make sure you have updated the folding factors in 'init_prune.py'. Please enter test id: "
read id
echo -e "Please enter dataset (CIFAR-10/SVHN/MNIST/RAILSEM):"
read dataset
echo "Start test $id"

if [ $dataset == 'MNIST' ]
then
	trainEpochs=50
	retrainEpochs=10
elif [ $dataset == 'CIFAR-10' ] || [ $dataset == 'SVHN' ]
then
	trainEpochs=200
	retrainEpochs=50
elif [ $dataset == 'COCO' ] || [ $dataset == 'RAILSEM' ]
then
	trainEpochs=35
	retrainEpochs=35
else
	echo -e "Please make sure that the dataset is one of (CIFAR-10/SVHN/MNIST/RAILSEM)."
	exit
fi



# cd models/${dataset}/scripts
# python bnn_pruning.py           # generate baseline --> pretrained_pruned.h5
# echo "finish bnn_pruning"
# cp pretrained_pruned.h5 ../pretrained_pruned.h5   # 提前一路径 ---> script/ ----> RAILSEM/
# cd ../../..
# python Binary.py ${dataset} True False True False True True False ${retrainEpochs} > output.txt  # retrain ='models/'+dataset+'/pretrained_pruned.h5'   --->   2_residuals.h5 

# # python Binary.py ${dataset} True False False False True True False ${1} > output.txt # without retraining

# mkdir -p models/${dataset}/pruned_bnn
# cp models/${dataset}/2_residuals.h5 models/${dataset}/pruned_bnn/pruned_bnn_${id}.h5
# cp output.txt models/${dataset}/pruned_bnn/pruned_bnn_${id}.txt
# cp models/${dataset}/2_residuals.h5 models/${dataset}/scripts/baseline_pruned.h5   # 2_residuals.h5  ---->  baseline_pruned.h5  
# #-------------------------Evaluate
# python Binary.py ${dataset} False False True False True True True ${retrainEpochs} > output.txt
# cp output.txt models/${dataset}/pruned_bnn/pruned_bnn_${id}_evaluation.txt
   

# cd models/${dataset}/scripts																		# dummy + baseline_pruned ----> pretrained_bin.h5
# python lutnet_init.py																
# cp pretrained_bin.h5 ../pretrained_bin.h5
# cd ../../..
# python Binary.py ${dataset} True False False True True False False ${trainEpochs} > output.txt      # retraining from pretrained_bin.h5         # weights_path='models/'+dataset+'/pretrained_bin.h5' # model.load_weights(weights_path)
# mkdir -p models/${dataset}/pruned_lutnet
# cp models/${dataset}/2_residuals.h5 models/${dataset}/pruned_lutnet/pruned_lutnet_${id}_BIN.h5
# cp output.txt models/${dataset}/pruned_lutnet/pruned_lutnet_${id}_BIN.txt
#-------------------------Evaluate
python Binary.py ${dataset} False False False True True False True ${trainEpochs} > output.txt 
cp output.txt models/${dataset}/pruned_lutnet/pruned_lutnet_${id}_BIN_evaluation.txt

#Progress:
#Software ---- REG

#Question 1
#LUT Regularizer --- train lutnet do we need regularizer?
#GPU Resource Exhausted due to size of tensor too large? (Largest Layer 6) 1024 - 2048  (4 --> 1) change to (Layer 5)

#Question 2
#Compilation Problem 

#Question 3
#Decode (Hardware or Software part)   +   Output to ram (Combine before or after write to RAM)

#Question 4   
#Concatenation FIFO (depth)

#Question 5
#Hardware; maxpooling & stride 1

#Question 6
#LUTNET (Not support in testbench provided by Marie)