#================================================================
#
#   Reading data output from hw
#
#================================================================

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
# import random 
# import time

import tensorflow as tf
# import tensorflow.keras as keras
# from tensorflow.keras.layers import Lambda

# import sys
# sys.path.insert(0, '..')
# from yolov3.utils import detect_image
# from yolov3.yolov4 import decode
# from binarization_utils import *
# from model_architectures import get_model

# 36 channels 

for i in range (3):
    if i == 0:
        OFM_file = open('channel_' + str(i) + ".txt", 'r')
        OFM = np.loadtxt(OFM_file)
        OFM.astype(np.float32)   

        OFMs = OFM
        print("OFM", i)
        print(OFMs.shape)
    # elif i == 1:
    else:
        OFM_file = open('channel_' + str(i) + ".txt", 'r')
        OFM = np.loadtxt(OFM_file)
        OFM.astype(np.float32)   

        OFMs = np.dstack([OFMs, OFM])
        print("OFM", i)
        print(OFMs.shape)


Output_tensor = tf.convert_to_tensor(OFMs) 
print(type(Output_tensor))
tf.print(Output_tensor)
print("-------------")
print(Output_tensor)
