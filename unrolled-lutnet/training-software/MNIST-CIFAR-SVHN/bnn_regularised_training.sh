#!/bin/bash

echo -e "Please enter dataset (CIFAR-10/SVHN/MNIST/RAILSEM/COCO):"
read dataset
echo "Start training bnn from scratch."

if [ $dataset == 'MNIST' ]
then
        trainEpochs=50
elif [ $dataset == 'RAILSEM' ]
then
        trainEpochs=36
elif [ $dataset == 'COCO' ]
then
        trainEpochs=5
elif [ $dataset == 'CIFAR-10' ] || [ $dataset == 'SVHN' ]
then
        trainEpochs=200
else
        echo -e "Please make sure that the dataset is one of (CIFAR-10/SVHN/MNIST/RAILSEM/COCO)."
        exit
fi

python Binary.py ${dataset} True True False False False True False ${trainEpochs} > output.txt

cp models/${dataset}/2_residuals.h5 models/${dataset}/scripts/baseline_reg.h5
cp output.txt models/${dataset}/scripts/baseline_reg.txt
echo -e "Finished training bnn from scratch. For LUTNet please run lutnet_training_script.sh."
echo -e "Not Evaluated Yet"
echo -e "Finished training bnn from scratch. Yolo training done. check if evaluation is running."
python Binary.py ${dataset} False True False False False True True ${trainEpochs} > output.txt
cp output.txt models/${dataset}/scripts/baseline_evaluation.txt
echo -e "Finished training bnn from scratch. Yolo evaluation done. Check output.txt."



# python Binary.py RAILSEM False True False False False True True 1 > output.txt