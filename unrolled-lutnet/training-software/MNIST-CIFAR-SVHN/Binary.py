#================================================================
#
#   File name   : Binary.py
#   Description : the main training script
#
#================================================================
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt
import matplotlib
import sys
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.datasets import cifar10,mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Lambda
#for tensorboard

# Core change, evaluation, train_class for Darknet, conv layer 25 Without Bias

'''
Updated for training in binary
'''
import os
print("-------------------------------------------------------------------")
print(tf.config.list_physical_devices('GPU'))
print("-------------------------------------------------------------------")
physical_devices = tf.config.list_physical_devices('GPU')
try:
  # Disable first GPU
  #tf.config.set_visible_devices(physical_devices[1:], 'GPU')
  tf.config.set_visible_devices(physical_devices, 'GPU')
  logical_devices = tf.config.list_logical_devices('GPU')
  # Logical device was not created for first GPU
  assert len(logical_devices) == len(physical_devices) - 1
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #yolo training

import sys
import tensorflow as tf
sys.path.insert(0, '..')
import shutil
from binarization_utils import *
from model_architectures import get_model
#yolo library
from yolov3.dataset import Dataset
from yolov3.yolov4 import compute_loss, decode, read_class_names
from yolov3.utils import load_yolo_weights
from yolov3.configs import *
from evaluate_mAP import get_mAP 

tf.config.experimental_run_functions_eagerly(True)
dataset = sys.argv[1]
Train = sys.argv[2] == 'True'
REG = sys.argv[3] == 'True'
Retrain = sys.argv[4] == 'True'
LUT = sys.argv[5] == 'True'
BINARY = sys.argv[6] == 'True'
trainable_means = sys.argv[7] == 'True'
Evaluate = sys.argv[8] == 'True'
epochs = int(sys.argv[9])

batch_size=100

print('Dataset is ', dataset)
print('Train is ', Train)					 #True Dummy: T     	   T	  T    T
print('REG is ', REG)					     #True Dummy: T		T  
print('Retrain is ', Retrain)				 #					       T	   
print('LUT is ', LUT)						 #	   Dummy: T			  	      T
print('BINARY is ', BINARY)					 #					       T      T
print('trainable_means is ', trainable_means)#True Dummy: T		T	   T
print('Evaluate is ', Evaluate)				 #						          T

if dataset == "RAILSEM" or dataset == "COCO": #initial yolo setting
	if YOLO_TYPE == "yolov4":
		Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
	if YOLO_TYPE == "yolov3":
		Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

def l2_reg(weight_matrix):
	return 5e-7 * K.sqrt(K.sum(K.abs(weight_matrix)**2))

def load_svhn(path_to_dataset):
	import scipy.io as sio
	train=sio.loadmat(path_to_dataset+'/train.mat')
	test=sio.loadmat(path_to_dataset+'/test.mat')
	extra=sio.loadmat(path_to_dataset+'/extra.mat')
	X_train=np.transpose(train['X'],[3,0,1,2])
	y_train=train['y']-1

	X_test=np.transpose(test['X'],[3,0,1,2])
	y_test=test['y']-1

	X_extra=np.transpose(extra['X'],[3,0,1,2])
	y_extra=extra['y']-1

	X_train=np.concatenate((X_train,X_extra),axis=0)
	y_train=np.concatenate((y_train,y_extra),axis=0)

	return (X_train,y_train),(X_test,y_test)

def load_yolo():
	if YOLO_TYPE == "yolov4":
		Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS
	if YOLO_TYPE == "yolov3":
		Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS
	#if TRAIN_YOLO_TINY: TRAIN_MODEL_NAME += "_Tiny"
	trainset = Dataset('train')
	testset = Dataset('test')
	
	return trainset, testset

def create_yolo_dataset(CLASS,resid_levels,LUT,BINARY,trainable_means,input_layer,model_name,evaluation):
    sys.stderr.write("TRAIN_FROM_SAVE_PATH: " + f'{TRAIN_FROM_SAVE_PATH}\n')
    sys.stderr.write("SAVE_PATH: "+ f'{SAVE_PATH}\n')
    sys.stderr.write("SAVE_PATH_EVAL: "+ f'{SAVE_PATH_EVAL}\n')
    sys.stderr.write("TRAIN_TRANSFER: "+ f'{TRAIN_TRANSFER}\n')
    sys.stderr.write("CUSTOM_WEIGHT: "+f'{CUSTOM_WEIGHT}\n')
    sys.stderr.write("FREEZE_CONV: "+f'{FREEZE_CONV}\n')

    NUM_CLASS = len(read_class_names(CLASS))
    sys.stderr.write("NUM_CLASS: "+f'{NUM_CLASS}\n')
    conv_tensors = get_model(model_name,resid_levels,LUT,BINARY,trainable_means,NUM_CLASS,input_layer,evaluation)
    def decode_layer(conv_tensors):
        output_tensors = []
        for i, conv_tensor in enumerate(conv_tensors):
            if i <= 1: 
                pred_tensor = decode(conv_tensor, NUM_CLASS, i)
                if not evaluation: output_tensors.append(conv_tensor)
                output_tensors.append(pred_tensor)
            # if i == 2:
            #     print("bn_4: shape", conv_tensor.shape, " ", conv_tensor)
        return output_tensors
    output_tensors = Lambda(decode_layer)(conv_tensors)

    return keras.Model(inputs=input_layer, outputs=output_tensors) # Darknet 

if dataset=="MNIST":
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	# convert class vectors to binary class matrices
	X_train = X_train.reshape(-1,784)
	X_test = X_test.reshape(-1,784)
	use_generator = False
elif dataset=="CIFAR-10":
	use_generator = True
	(X_train, y_train), (X_test, y_test) = cifar10.load_data()
elif dataset=="SVHN":
	use_generator = True
	(X_train, y_train), (X_test, y_test) = load_svhn('./svhn_data')
elif dataset == "RAILSEM" or dataset == "COCO":
	trainset, testset = load_yolo()
else:
	raise("dataset should be one of the following: [MNIST, CIFAR-10, SVHN].")

if dataset != "RAILSEM" and dataset != "COCO":
	X_train=X_train.astype(np.float32)
	X_test=X_test.astype(np.float32)
	Y_train = to_categorical(y_train, 10)
	Y_test = to_categorical(y_test, 10)
	X_train /= 255
	X_test /= 255
	X_train=2*X_train-1
	X_test=2*X_test-1

	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')

if Train:
    if not(os.path.exists('models')):
        os.mkdir('models')
    if not(os.path.exists('models/'+dataset)):
        os.mkdir('models/'+dataset)
        
    for resid_levels in range(2,3): #range(1,4):    
        print('training with', resid_levels,'levels')
        #sess=tf.compat.v1.Session()
        if dataset == "RAILSEM" or dataset == "COCO":
            input_layer = keras.Input(shape=(608,608,3), name="img")
            model = create_yolo_dataset(TRAIN_CLASSES,resid_levels,LUT,BINARY,trainable_means,input_layer,'yolov3_tiny',evaluation=False)
            Darknet = create_yolo_dataset(YOLO_COCO_CLASSES,resid_levels,LUT,BINARY,trainable_means,input_layer,'yolov3_tiny',evaluation=False) #Darknet = create_yolo_dataset(TRAIN_CLASSES,resid_levels,LUT,BINARY,trainable_means,input_layer,'yolov3_tiny_double_channel',evaluation=False)

            if TRAIN_TRANSFER: # model is extract from here  YOLOv3()  loading initial weight
                if CUSTOM_WEIGHT:
                    Darknet.load_weights(SAVE_PATH) #use custom weights
                else:
                    load_yolo_weights(Darknet, Darknet_weights) #use darknet weights

            if TRAIN_TRANSFER and not TRAIN_FROM_CHECKPOINT: #loading initial weight
                for i, l in enumerate(Darknet.layers):
                    layer_weights = l.get_weights()
                    if layer_weights != []:
                        try:
                            model.layers[i].set_weights(layer_weights)
                        except:
                            print("skipping", model.layers[i].name)

            if TRAIN_FROM_SAVE_PATH: # Yolo Model directly with .h5 weights
                model.load_weights(SAVE_PATH)

            if FREEZE_CONV:
                print("Number of layers in the base model: ", len(model.layers))
                for layer in model.layers:
                    print(layer.name)
                    #if layer.name != "binary_conv_bias_1" and layer.name != "binary_conv_bias":
                    if layer.name != "binary_conv_12" and layer.name != "binary_conv_9":         # Without Bias
                        layer.trainable = False

        else:
            model=get_model(dataset,resid_levels,LUT,BINARY,trainable_means)
        
        #Model Inform
        
        print("No_bias")
        #print(Darknet.summary())
        for i, w in enumerate(model.weights): print(i, w.name)
        print("With_bias")
        print(model.summary())
        for i, w in enumerate(model.weights): print(i, w.name)
        print("DONE")
        #tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#----------------------------------------------------------------------------------------------------------
#LUTNet Model Config
#----------------------------------------------------------------------------------------------------------
		#gather all binary dense and binary convolution layers:
        binary_layers=[]
        for l in model.layers:
            if isinstance(l,binary_dense) or isinstance(l,binary_conv):
                binary_layers.append(l)

        #gather all residual binary activation layers:
        resid_bin_layers=[]
        for l in model.layers:
            if isinstance(l,Residual_sign):
                resid_bin_layers.append(l)
        lr=0.01
        decay=1e-6

        if Retrain:
            weights_path='models/'+dataset+'/pretrained_pruned.h5'
            model.load_weights(weights_path)
        elif LUT and not REG:
            weights_path='models/'+dataset+'/pretrained_bin.h5'
            model.load_weights(weights_path)
        elif REG and not LUT:
            tmp = []
            vars = tf.compat.v1.trainable_variables() 

            if dataset == 'CIFAR-10' or dataset == 'SVHN':
                for v in vars:
                    if (('binary_conv_6' in v.name and 'Variable_' in v.name) and 'Variable_' in v.name): #  # reg applied onto conv_6 only
                        tmp.append(tf.nn.l2_loss(v) *  (5e-7))
            elif dataset == 'MNIST':
                for v in vars:
                    if (('binary_dense_' in v.name and 'binary_dense_1' not in v.name) and 'Variable_' in v.name):
                        tmp.append(tf.nn.l2_loss(v) *  (5e-7))
                
                # lossL2 = tf.math.add_n([ tf.nn.l2_loss(v) for v in vars
                # 	if ('binary_dense_' in v.name and 'binary_dense_1' not in v.name) and 'Variable_' in v.name ]) * 5e-7 # reg applied onto conv_6 only

            # elif dataset == "RAILSEM" or dataset == "COCO":
            #     for v in vars:
            #         if (('binary_conv' in v.name and 'Variable_' in v.name) and 'Variable_' in v.name): #  # reg applied onto conv_6 only
            #             tmp.append(tf.nn.l2_loss(v) *  (5e-7))
            #         elif (('binary_conv_' in v.name and 'Variable_' in v.name) and 'Variable_' in v.name): #  # reg applied onto conv_6 only
            #             tmp.append(tf.nn.l2_loss(v) *  (5e-7))
            try:
                lossL2 = tf.math.add_n(tmp)
                print(" Regularization Ready ")
            except:
                print(" Regularization Fail")
#----------------------------------------------------------------------------------------------------------------
        if dataset == "RAILSEM" or dataset == "COCO":
            #railsem & COCO
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
            
            writer = tf.summary.create_file_writer(train_log_dir)
            validate_writer = tf.summary.create_file_writer(test_log_dir)

            steps_per_epoch = len(trainset)
            global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
            warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
            total_steps = TRAIN_EPOCHS * steps_per_epoch

            optimizer = tf.keras.optimizers.Adam()
            #-------------------------------------------------------------------------------------------		
            def train_step(image_data, target, epoch):
                with tf.GradientTape() as tape:
                    pred_result = model(image_data, training=True)
                    #  print("<--------------------train_step MODEL OUTPUT----------------->")
                    #  print(pred_result)

                    giou_loss=conf_loss=prob_loss=0
                    grid = 3 if not TRAIN_YOLO_TINY else 2
                    for i in range(grid):
                        conv, pred = pred_result[i*2], pred_result[i*2+1]
                        loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                        
                        giou_loss += loss_items[0]
                        # print("giou_loss")
                        # print(giou_loss)
                        conf_loss += loss_items[1]
                        # print("conf_loss")
                        # print(conf_loss)
                        prob_loss += loss_items[2]
                        # print("prob_loss")
                        # print(prob_loss)
                    

                    # Adding l2 regularizer
                    if LUT and not REG:
                        lossL2 = 0   
                    else: 
                        tmp = []
                        vars = model.trainable_variables
                        print(len(vars))
                        for v in vars:
                            if (('binary_conv' in v.name and 'Variable_' in v.name) and 'Variable_' in v.name): #  # reg applied onto conv_6 only
                                
                                tmp.append(tf.nn.l2_loss(v) *  (5e-7))
                            elif (('binary_conv_' in v.name and 'Variable_' in v.name) and 'Variable_' in v.name): #  # reg applied onto conv_6 only
                            
                                tmp.append(tf.nn.l2_loss(v) *  (5e-7))
                        if(len(vars) != 0):
                            lossL2 = tf.math.add_n(tmp)
                            print(" Regularization Ready ")
                        else:
                            lossL2 = 0
                    
                    # apply optimizer on gradient
                    total_loss = giou_loss + conf_loss + prob_loss + lossL2
                    print("total_loss : ", total_loss, " giou_loss : ", giou_loss, " conf_loss :", conf_loss, " prob_loss : ", prob_loss)
                    gradients = tape.gradient(total_loss, model.trainable_variables)
                    # print("<----------------Gradient START-------------->")
                    # for grad in gradients: 
                    #     print(grad)
                    # print("<----------------Gradient END-------------->")
                    # clear_gradients = [tf.constant(0) if grad is None else grad for grad in gradients]
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    # update learning rate
                    # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
                    global_steps.assign_add(1)
                    # Update ref by adding value to it.
                    if global_steps < warmup_steps:# and not TRAIN_TRANSFER:
                        lr = global_steps / warmup_steps * TRAIN_LR_INIT
                    else:
                        lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
                            (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
                    optimizer.lr.assign(lr.numpy())

                    # writing summary data
                    with writer.as_default():
                        #Writes simple numeric values for later analysis in TensorBoard. 
                        #Writes go to the current default summary writer. 
                        #Each summary point is associated with an integral step value. 
                        #This enables the incremental logging of time series data. A common usage of this API is to log loss during training to produce a loss curve

                        tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                        tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                        tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
                        tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                        tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
                    writer.flush()

                return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()
            #-------------------------------------------------------------------------------------------
            def validate_step(image_data, target, model):
                with tf.GradientTape() as tape:
                    pred_result = model(image_data, training=False)
                    giou_loss=conf_loss=prob_loss=0

                    # optimizing process
                    grid = 3 if not TRAIN_YOLO_TINY else 2
                    for i in range(grid):
                        conv, pred = pred_result[i*2], pred_result[i*2+1]
                        loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                        giou_loss += loss_items[0]
                        conf_loss += loss_items[1]
                        prob_loss += loss_items[2]

                    total_loss = giou_loss + conf_loss + prob_loss
                    
                return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()
            #-------------------------------------------------------------------------------------------
            best_val_loss = 1000 # should be large at start
            for epoch in range(TRAIN_EPOCHS):
                if epoch == 3 and FREEZE_CONV == True:
                    fine_tune_at = 31
                    print("Number of layers in the base model: ", len(model.layers))
                    for layer in model.layers:
                        layer.trainable = True
                    for layer in model.layers[:fine_tune_at]:
                        layer.trainable = False

                if epoch == 5 and FREEZE_CONV == True:
                    for layer in model.layers:
                        layer.trainable = True
                
                for image_data, target in trainset:
                    # image_data = K.constant(image_data)
                    results = train_step(image_data, target, epoch)
                    cur_step = results[0]%steps_per_epoch
                    if(cur_step % 100 == 1):
                        # print("LOST")
                        # print(results[1])
                        # print(results[2])
                        # print(results[3])
                        # print(results[4])
                        # print(results[5])
                        print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}"
                            .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5]))
                        sys.stderr.write("epoch:{:2.0f} step:{:10.0f}/{}, total_loss:{:7.2f} \n" .format(epoch, cur_step, steps_per_epoch,  results[5]))
                        

                if len(testset) == 0:
                    print("configure TEST options to validate model")
                    model.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
                    continue
            
                count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
                for image_data, target in testset:
                    results = validate_step(image_data, target,model)
                    count += 1
                    giou_val += results[0]
                    conf_val += results[1]
                    prob_val += results[2]
                    total_val += results[3]

                # writing validate summary data
                # with validate_writer.as_default():
                # 	tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
                # 	tf.summary.scalar("validate_loss/giou_val", giou_val/count, step=epoch)
                # 	tf.summary.scalar("validate_loss/conf_val", conf_val/count, step=epoch)
                # 	tf.summary.scalar("validate_loss/prob_val", prob_val/count, step=epoch)
                # validate_writer.flush()
                    with validate_writer.as_default():
                        tf.summary.scalar("loss/total_loss", total_val/count, step=epoch)
                        tf.summary.scalar("loss/giou_loss", giou_val/count, step=epoch)
                        tf.summary.scalar("loss/conf_loss", conf_val/count, step=epoch)
                        tf.summary.scalar("loss/prob_loss", prob_val/count, step=epoch)
                    validate_writer.flush()
                    
                print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
                    format(giou_val/count, conf_val/count, prob_val/count, total_val/count))

                if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
                    save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"_val_loss_{:7.2f}".format(total_val/count))
                    model.save_weights(save_directory)
                if TRAIN_SAVE_BEST_ONLY and best_val_loss>total_val/count:
                    save_directory='models/'+dataset+'/'+str(resid_levels)+'_residuals.h5'
                    #save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
                    model.save_weights(save_directory)
                    best_val_loss = total_val/count
                if not TRAIN_SAVE_BEST_ONLY and not TRAIN_SAVE_CHECKPOINT:
                    save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
                    #save_directory='models/'+dataset+'/'+str(resid_levels)+'_residuals.h5'
                    model.save_weights(save_directory)                  
        else:
            opt = keras.optimizers.Adam(lr=lr,decay=decay)#SGD(lr=lr,momentum=0.9,decay=1e-5)
            model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

            weights_path='models/'+dataset+'/'+str(resid_levels)+'_residuals.h5'
            cback=keras.callbacks.ModelCheckpoint(weights_path, monitor='val_accuracy', save_best_only=True)

            if use_generator:
                if dataset=="CIFAR-10":
                    horizontal_flip=True
                if dataset=="SVHN":
                    horizontal_flip=False
                datagen = ImageDataGenerator(
                    width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=horizontal_flip)  # randomly flip images
                if keras.__version__[0]=='2':
                    history=model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size),steps_per_epoch=X_train.shape[0]/batch_size,
                    nb_epoch=epochs,validation_data=(X_test, y_test),verbose=2,callbacks=[cback])
                if keras.__version__[0]=='1':
                    history=model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size), samples_per_epoch=X_train.shape[0], 
                    nb_epoch=epochs, verbose=2,validation_data=(X_test,y_test),callbacks=[cback])

            else:
                if keras.__version__[0]=='2':
                    history=model.fit(X_train, y_train,batch_size=batch_size,validation_data=(X_test, y_test), verbose=2,epochs=epochs,callbacks=[cback])
                if keras.__version__[0]=='1':
                    history=model.fit(X_train, y_train,batch_size=batch_size,validation_data=(X_test, y_test), verbose=2,nb_epoch=epochs,callbacks=[cback])
            dic={'hard':history.history}
            foo=open('models/'+dataset+'/history_'+str(resid_levels)+'_residuals.pkl','wb')
            pickle.dump(dic,foo)
            foo.close()

if Evaluate:
    for resid_levels in range(2,3):
        if BINARY:
            weights_path = SAVE_PATH_EVAL
        else:
            weights_path='models/'+dataset+'/'+str(resid_levels)+'_residuals.h5'
        #weights_path='models/'+dataset+'/'+'baseline_reg_4k.h5'
        if dataset == "RAILSEM" or dataset == "COCO":
                # measure mAP of trained custom model
            try:
                input_layer = keras.Input(shape=(608,608,3), name="img")
                mAP_model = create_yolo_dataset(TRAIN_CLASSES,resid_levels,LUT,BINARY,trainable_means,input_layer,'yolov3_tiny', evaluation=True)
                #mAP_model.summary()
                mAP_model.load_weights(weights_path) # use keras weights
                print("Weight Path: ", weights_path)
                get_mAP(mAP_model, testset, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD)
            
            except UnboundLocalError:
                print("You don't have saved model weights to measure mAP, check TRAIN_SAVE_BEST_ONLY and TRAIN_SAVE_CHECKPOINT lines in configs.py")
        else:
            model=get_model(dataset,resid_levels,LUT,BINARY,trainable_means)
            model.load_weights(weights_path)
            opt = keras.optimizers.Adam()
            model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
            #model.summary()
            


