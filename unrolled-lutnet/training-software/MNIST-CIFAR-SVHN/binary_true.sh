#================================================================
#
#   File name   : binary_true.sh
#   Created date: 2020-08-30
#   Description : scripts that perform binaried model training and evaluation
#
#   Output File: 
#       pruned_bnn/pruned_bnn_${id}.h5                       h5 with binarized weight without conversion
#       scripts   /baseline_pruned.h5
#       ${dataset}   /pretrained_pruned.h5                  h5 right after prunnubg
#       
#       scripts   /baseline_reg.h5.h5                    h5 with binarized weight with conversion
#       scripts   /baseline_reg_conversion.h5            h5 with binarized weight with conversion
#================================================================

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
	trainEpochs=70
	retrainEpochs=50
else
	echo -e "Please make sure that the dataset is one of (CIFAR-10/SVHN/MNIST/RAILSEM)."
	exit
fi

cd models/${dataset}/scripts
python bnn_pruning.py           # generate baseline --> pretrained_pruned.h5
echo "finish bnn_pruning"
cp pretrained_pruned.h5 ../pretrained_pruned.h5   # 提前一路径 ---> script/ ----> ${dataset}/
cd ../../..
echo -e "Check config !"
echo -e "Check config !"
echo -e "Check config !"
echo -e "Please ensure you do have the correct yolov3/yolov4.py file with 0 area been handled"
python Binary.py ${dataset} True False True False True True False 35 > output.txt  # retrain ='models/'+dataset+'/pretrained_pruned.h5'   --->   2_residuals.h5 


mkdir -p models/${dataset}/pruned_bnn
cp models/${dataset}/2_residuals.h5 models/${dataset}/pruned_bnn/pruned_bnn_${id}.h5
cp output.txt models/${dataset}/pruned_bnn/pruned_bnn_${id}.txt
cp models/${dataset}/2_residuals.h5 models/${dataset}/scripts/baseline_pruned.h5   # 2_residuals.h5  ---->  baseline_pruned.h5  

echo -e "Training Finished"
echo -e "Do evaluation !"
echo -e "Please enter evaluation mode:"
read mode
echo "Start training bnn from scratch."

if [ $mode == 'full' ]
then
        echo -e "ok"
elif [ $mode == 'half' ]
then
        echo -e "Change config.py weight path to scripts/baseline_reg.h5"
        echo -e "Change Binary.py evaluation into false in creating mAP model"
else
        echo -e "Please make sure that the evaluation mode is one of (full/half)."
        exit
fi

cd models/${dataset}
python conversion.py
cp baseline_reg.h5 scripts/baseline_reg_conversion.h5
echo -e "COMPLETE! conversion now running evaluation"
cd ../../
echo -e "WARNING: CHECK Config path: "
echo -e "Make sure the weight path is correct"

python Binary.py ${dataset} False False True False True True True 1 ${mode}> output.txt
cp output.txt models/${dataset}/scripts/baseline_evaluation_pruned.txt

echo -e "evaluation finished result in models/${dataset}/scripts/baseline_evaluation_pruned.txt"



# ""
# Train = sys.argv[2] == 'True'
# REG = sys.argv[3] == 'True'
# Retrain = sys.argv[4] == 'True'
# LUT = sys.argv[5] == 'True'
# BINARY = sys.argv[6] == 'True'
# trainable_means = sys.argv[7] == 'True'
# Evaluate = sys.argv[8] == 'True
# """

# mkdir -p models/${dataset}/pruned_bnn
# cp models/${dataset}/2_residuals.h5 models/${dataset}/pruned_bnn/pruned_bnn_${id}.h5
# cp output.txt models/${dataset}/pruned_bnn/pruned_bnn_${id}.txt
# cp models/${dataset}/2_residuals.h5 models/${dataset}/scripts/baseline_pruned.h5  