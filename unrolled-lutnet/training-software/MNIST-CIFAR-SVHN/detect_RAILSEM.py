#================================================================
#
#   File name   : detect_mnist.py
#   Description : RAILSEM object detection example
#
#================================================================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import numpy as np
import random 
import time
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Lambda

import sys
np.set_printoptions(threshold=sys.maxsize)
sys.path.insert(0, '..')
from yolov3.utils import detect_image
from yolov3.yolov4 import decode
from binarization_utils import *
from model_architectures import get_model

# python3 detect_RAILSEM.py 1 1 1 1 1 1 1 1 1 > detect_result.txt
#------------------------------------TESTING CONFIG-----------------------------------
TRAIN_CLASSES               = "models/RAILSEM/mnist.names"
YOLO_FRAMEWORK              = "tf" # "tf" or "trt"
YOLO_INPUT_SIZE             = 608
resid_levels = 2
LUT = False
BINARY = True
trainable_means = True
#------------------------------------TESTING CONFIG-----------------------------------

def image_preprocess(image, target_size, gt_boxes=None):
    ih, iw    = target_size
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)                 # 608/1080, 608/1920
    nw, nh  = int(scale * w), int(scale * h) # 608, 608/1920*1080=342
    image_resized = cv2.resize(image, (nw, nh)) 

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)          # 608, 608, 3
    dw, dh = (iw - nw) // 2, (ih-nh) // 2                               # 0, 342
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized                  # 
    image_paded = image_paded / 255
    # img = Image.fromarray(image_paded)
    # img.save('myimg.jpeg')

    if gt_boxes is None:
        return image_paded

    else:
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes

def create_yolo_model(CLASS,resid_levels,LUT,BINARY,trainable_means,input_layer,model_name,evaluation):
    NUM_CLASS = 7
    conv_tensors = get_model(model_name,resid_levels,LUT,BINARY,trainable_means,NUM_CLASS,input_layer,evaluation)
    def decode_layer(conv_tensors):
        output_tensors = []
        for i, conv_tensor in enumerate(conv_tensors):
            if i <= 1: 
                pred_tensor = decode(conv_tensor, NUM_CLASS, i)
                if not evaluation: output_tensors.append(conv_tensor)
                output_tensors.append(pred_tensor)
            # if i == 2:
            #     print("bn_1: shape", conv_tensor.shape, " ", conv_tensor)
        return output_tensors
    output_tensors = Lambda(decode_layer)(conv_tensors)

    return keras.Model(inputs=input_layer, outputs=output_tensors) # Darknet    


# rs00206
# ID = random.randint(0, 200)
# #label_txt = "MNIST-CIFAR-SVHN/models/RAILSEM/mnist/mnist_test.txt"
# label_txt = "models/RAILSEM/mnist_test.txt"
# image_info = open(label_txt).readlines()[ID].split()
# image_path = image_info[0]

#laoding img for model input
image_path = "/mnt/storage/hwthw20/LUTNet2/LUTNet/unrolled-lutnet/training-software/MNIST-CIFAR-SVHN/models/RAILSEM/mnist/rs01269.jpg"
input_size = 608
original_image      = cv2.imread(image_path)
original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image      = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)


input_layer = keras.Input(shape=(608,608,3), name="img")
yolo = create_yolo_model(TRAIN_CLASSES,resid_levels,LUT,BINARY,trainable_means,input_layer,'yolov3_tiny',evaluation=True)
for i in range(57):
    print("layer index: ", i , " layer name " , yolo.layers[i].name)

#yolo.load_weights(f"./checkpoints/{TRAIN_MODEL_NAME}") # use keras weights
yolo.load_weights(f"models/RAILSEM/scripts/baseline_reg.h5") # float32 
# detect_image(yolo, image_path, "RAILSEM_test.jpg", input_size=YOLO_INPUT_SIZE, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255,0,0))


aux_model_bin_conv_0 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[1].output)
aux_model_bnorm_0 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[2].output)
aux_model_res_0_bit_1 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[4].output[0])
aux_model_res_0_bit_2 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[4].output[1])


aux_model_bin_conv_1 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[5].output)
aux_model_bnorm_1 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[6].output)
aux_model_res_1_bit_1 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[8].output[0])
aux_model_res_1_bit_2 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[8].output[1])

# aux_model_bin_conv_2 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[3].output)
# aux_model_res_2_bit_1 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[6].output[0])
# aux_model_res_2_bit_2 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[6].output[1])

# aux_model_bin_conv_3 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[7].output)
# aux_model_res_3_bit_1 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[9].output[0])
# aux_model_res_3_bit_2 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[9].output[1])

# aux_model_res_4_bit_1 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[13].output[0])
# aux_model_res_4_bit_2 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[13].output[1])

# aux_model_res_5_bit_1 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[17].output[0])
# aux_model_res_5_bit_2 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[17].output[1])

# aux_model_res_6_bit_1 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[20].output[0])
# aux_model_res_6_bit_2 = tf.keras.Model(inputs=yolo.inputs, outputs=yolo.layers[20].output[1])
# print("SHAPE")
# x= aux_model_bin_conv_0.predict(image_data)
# print(x.shape)
# print("SHAPE1")
# print(aux_model_bnorm_0.predict(image_data).shape)

# print(aux_model_res_0_bit_1.predict(image_data).shape)
# print(aux_model_res_0_bit_2.predict(image_data).shape)


# Access intermediate outputs of the original model
bin_conv_0 = np.transpose(aux_model_bin_conv_0.predict(image_data), (0,3,1,2))
bnorm_0 = np.transpose(aux_model_bnorm_0.predict(image_data), (0,3,1,2))

residual_sign_0_bit_1 = np.transpose(aux_model_res_0_bit_1.predict(image_data), (0,3,1,2))
residual_sign_0_bit_2 = np.transpose(aux_model_res_0_bit_2.predict(image_data), (0,3,1,2))

bin_conv_1 = np.transpose(aux_model_bin_conv_1.predict(image_data), (0,3,1,2))
bnorm_1 = np.transpose(aux_model_bnorm_1.predict(image_data), (0,3,1,2))

residual_sign_1_bit_1 = np.transpose(aux_model_res_1_bit_1.predict(image_data), (0,3,1,2))
residual_sign_1_bit_2 = np.transpose(aux_model_res_1_bit_2.predict(image_data), (0,3,1,2))

# bin_conv_2 = np.transpose(aux_model_bin_conv_2.predict(image_data), (0,3,1,2))			
# residual_sign_2_bit_1 = np.transpose(aux_model_res_2_bit_1.predict(image_data), (0,3,1,2))
# residual_sign_2_bit_2 = np.transpose(aux_model_res_2_bit_2.predict(image_data), (0,3,1,2))

 

print("Input Image shape: ", image_data.shape, "Input Image:", image_data)

output_path = "models_output"
if not(os.path.exists(output_path)): os.mkdir(output_path)
# Save intermediate outputs
for ch in range(16):
    np.savetxt(output_path + '/bin_conv_0_ch_'+str(ch)+'.txt', bin_conv_0[0][ch][:][:])
    np.savetxt(output_path + '/bnorm_0_ch_'+str(ch)+'.txt', bnorm_0[0][ch][:][:])
    np.savetxt(output_path + '/residual_sign_0_bit_1_ch_'+str(ch)+'.txt', residual_sign_0_bit_1[0][ch][:][:])
    np.savetxt(output_path + '/residual_sign_0_bit_2_ch_'+str(ch)+'.txt', residual_sign_0_bit_2[0][ch][:][:])

for ch in range(32):
    np.savetxt(output_path + '/bin_conv_1_ch_'+str(ch)+'.txt', bin_conv_1[0][ch][:][:])
    np.savetxt(output_path + '/bnorm_1_ch_'+str(ch)+'.txt', bnorm_1[0][ch][:][:])
    np.savetxt(output_path + '/residual_sign_1_bit_1_ch_'+str(ch)+'.txt', residual_sign_1_bit_1[0][ch][:][:])
    np.savetxt(output_path + '/residual_sign_1_bit_2_ch_'+str(ch)+'.txt', residual_sign_1_bit_2[0][ch][:][:])

# logger.info("with %d residuals, test loss was %0.4f, test accuracy was %0.4f"%(resid_levels,score[0],score[1]))
# print("with %d residuals, test loss was %0.4f, test accuracy was %0.4f"%(resid_levels,score[0],score[1]))