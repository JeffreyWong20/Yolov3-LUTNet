#LUTNet Latest
from ast import Lambda
from pickle import FALSE, TRUE
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Convolution2D, Activation, Flatten, MaxPooling2D,Input,Dropout,GlobalAveragePooling2D,Lambda 
# from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
# from importlib import import_module
# from importlib import import_module
# MNIST_CIFAR_SVHN = import_module('MNIST-CIFAR-SVHN')
# from MNIST_CIFAR_SVHN.yolov3.yolov4 import BatchNormalization
from tensorflow.python.framework import ops
from binarization_utils import *


'''
binarization_utils with rounding to 0.5 operation
'''


#tf.compat.v1.disable_eager_execution()
class BatchNormalization(BatchNormalization):
    # "Frozen state" and "inference mode" are two separate concepts.
    # `layer.trainable = False` is to freeze the layer, so the layer will use
    # stored moving `var` and `mean` in the "inference mode", and both `gama`
    # and `beta` will not be updated !
	def call(self, x, training=False):
		if not training:
			training = tf.constant(False)
		# print("<------------within batch Normalization---------------->")
		# print("Input")
		# print(x)
		training = tf.logical_and(training, self.trainable)
		# print("Output")
		result = super().call(x, training)
		# print(result)
		# print("end of batch normalization")
		return result


batch_norm_eps=1e-4
batch_norm_alpha=0.1#(this is same as momentum)


def get_model(dataset,resid_levels,LUT,BINARY,trainable_means,NUM_CLASS=0,input_layer = keras.Input(shape=(608,608,3)),evaluate=False):
	if dataset=='MNIST':
		model = Sequential()
		model.add(binary_dense(levels=resid_levels,n_in=784,n_out=256,input_shape=[784],first_layer=True,BINARY=BINARY))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=256,LUT=LUT,BINARY=BINARY))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=256,LUT=LUT,BINARY=BINARY))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=256,LUT=LUT,BINARY=BINARY))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=10,LUT=LUT,BINARY=BINARY))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Activation('softmax'))
	
	elif dataset=="CIFAR-10" or dataset=="SVHN":
		model=Sequential()
		#0
		model.add(binary_conv(evaluate=evaluate,pruning_prob=0.1,nfilters=64,ch_in=3,k=3,padding='valid',input_shape=[32,32,3],first_layer=True,BINARY=BINARY))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means))
		#1
		model.add(binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=64,ch_in=64,k=3,padding='valid',LUT=False,BINARY=BINARY))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		model.add(Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means))
		#2
		model.add(binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.2,nfilters=128,ch_in=64,k=3,padding='valid',LUT=False,BINARY=BINARY))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means))
		#3
		model.add(binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.2,nfilters=128,ch_in=128,k=3,padding='valid',LUT=False,BINARY=BINARY))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		model.add(Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means))
		#4
		model.add(binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.3,nfilters=256,ch_in=128,k=3,padding='valid',LUT=False,BINARY=BINARY))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means))
		#5
		model.add(binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.3,nfilters=256,ch_in=256,k=3,padding='valid',LUT=LUT,BINARY=BINARY))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(my_flat())
		model.add(Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means))
		model.add(binary_dense(levels=resid_levels,pruning_prob=0.8,n_in=int(model.output.get_shape()[2]),n_out=512,LUT=False,BINARY=BINARY))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means))
		model.add(binary_dense(levels=resid_levels,pruning_prob=0.8,n_in=int(model.output.get_shape()[2]),n_out=512,LUT=False,BINARY=BINARY))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means))
		model.add(binary_dense(levels=resid_levels,pruning_prob=0.5,n_in=int(model.output.get_shape()[2]),n_out=10,LUT=False,BINARY=BINARY))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps))
		model.add(Activation('softmax'))
	
	elif dataset =="yolov3_tiny" :
		#0
		input_data = binary_conv(evaluate=evaluate,pruning_prob=0.1,nfilters=16,ch_in=3,k=3,padding='same',input_shape=[608,608,3],first_layer=True,BINARY=BINARY,layer_index=0)(input_layer)
		input_data_bn_0 = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)		
		
		input_data = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data_bn_0)		
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=0, LUT = LUT)(input_data)
		#1
		input_data = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=32,ch_in=16,k=3,padding='same',LUT=False,BINARY=BINARY,layer_index=1)(input_data)
		input_data_bn_1 = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)
		input_data	= MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data_bn_1)
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=1, LUT = LUT)(input_data)
		#2
		input_data = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=64,ch_in=32,k=3,padding='same',LUT=False,BINARY=BINARY,layer_index=2)(input_data)
		input_data_bn_2 = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)
		input_data	= MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data_bn_2)
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=2, LUT = LUT)(input_data)
		#3
		input_data = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=128,ch_in=64,k=3,padding='same',LUT=False,BINARY=BINARY,layer_index=3)(input_data)
		input_data_bn_3 = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)
		input_data	= MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data_bn_3)
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=3, LUT = LUT)(input_data)
		#4
		input_data = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=256,ch_in=128,k=3,padding='same',LUT=False,BINARY=BINARY,layer_index=4)(input_data)
		input_data_bn_4 = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)

		Resid_layer_1 = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=4, LUT = LUT)
		input_data = Resid_layer_1(input_data_bn_4)
		route_1 = input_data

		input_data_1, input_data_2 = tf.split(input_data, num_or_size_splits=2, axis=0)
		input_data_1=tf.squeeze(input_data_1,0)
		input_data_2=tf.squeeze(input_data_2,0)



		input_data = tf.math.add(input_data_1, input_data_2)
		input_data	= MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data)

		means = Resid_layer_1.means

		Resid_layer_rover = Residual_sign_recover(BINARY=BINARY,evaluate=evaluate,means=means, levels=resid_levels,trainable=False,layer_index=4, LUT = LUT)
		input_data=Resid_layer_rover(input_data)

		
		# input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means)(input_data)
		# route_1 = input_data
		# input_data	= MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data)
		#5
		input_data = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=512,ch_in=256,k=3,padding='same',LUT=LUT,BINARY=BINARY,layer_index=5)(input_data)
		input_data = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)
		input_data	= MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding="same")(input_data)
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means, layer_index=5, LUT = LUT)(input_data)
	#---------------------------------------------------------------------------------------------------------------
		#6
		conv = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=1024,ch_in=512,k=3,padding='same',LUT=False,BINARY=BINARY,layer_index=6)(input_data)
		conv = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(conv)
		conv = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=6, LUT = LUT )(conv)
		#7
		conv = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=256,ch_in=1024,k=1,padding='same',LUT=False,BINARY=BINARY,layer_index=7)(conv)
		conv = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(conv)
		conv = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=7, LUT = LUT)(conv)
		#8
		conv_lobj_branch = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=512,ch_in=256,k=3,padding='same',LUT=False,BINARY=BINARY,layer_index=8)(conv)
		conv_lobj_branch = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(conv_lobj_branch)
		conv_lobj_branch = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=8, LUT = LUT)(conv_lobj_branch)
		#9 Output
		conv_lbbox = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=3*(NUM_CLASS + 5),ch_in=512,k=1,padding='same',LUT=False,BINARY=BINARY,layer_index=9)(conv_lobj_branch)
		#10
		conv = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=128,ch_in=256,k=1,padding='same',LUT=False,BINARY=BINARY,layer_index=10)(conv)
		conv = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(conv)
		Resid_layer_2 = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=9, LUT = LUT)
		conv = Resid_layer_2(conv)

		conv_1, conv_2 = tf.split(conv, num_or_size_splits=2, axis=0)
		conv_1=tf.squeeze(conv_1,0)  
		conv_2=tf.squeeze(conv_2,0)

		conv_1 = tf.image.resize(conv_1, (conv_1.shape[1] * 2, conv_1.shape[2] * 2), method='nearest')
		conv_2 = tf.image.resize(conv_2, (conv_2.shape[1] * 2, conv_2.shape[2] * 2), method='nearest')

		conv_1 = tf.expand_dims(conv_1, axis=0)
		conv_2 = tf.expand_dims(conv_2, axis=0)
		conv = tf.concat([conv_1, conv_2], axis=0)
		
		# conv = tf.image.resize(conv, (conv.shape[1] * 2, conv.shape[2] * 2), method='nearest')
		conv = tf.concat([conv, route_1], axis=-1)
		#11
		conv_mobj_branch = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=256,ch_in=384,k=3,padding='same',LUT=False,BINARY=BINARY,layer_index=11)(conv)	
		conv_mobj_branch = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(conv_mobj_branch)
		conv_mobj_branch = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=10, LUT = LUT)(conv_mobj_branch)
		#12 Output
		conv_mbbox = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=3*(NUM_CLASS + 5),ch_in=256,k=1,padding='same',LUT=False,BINARY=BINARY,layer_index=12)(conv_mobj_branch)
		
		return [conv_mbbox, conv_lbbox, input_data_bn_1]

	elif dataset =="yolov3_tiny_double_channel_v2" :
		#0
		input_data = binary_conv(evaluate=evaluate,pruning_prob=0.1,nfilters=32,ch_in=3,k=3,padding='same',input_shape=[608,608,3],first_layer=True,BINARY=BINARY,layer_index=0)(input_layer)
		input_data = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)		
		input_data = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data)		
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=0)(input_data)
		#1
		input_data = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=64,ch_in=32,k=3,padding='same',LUT=False,BINARY=BINARY,layer_index=1)(input_data)
		input_data = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)
		input_data	= MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data)
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=1)(input_data)
		#2
		input_data = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=128,ch_in=64,k=3,padding='same',LUT=False,BINARY=BINARY,layer_index=2)(input_data)
		input_data = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)
		input_data	= MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data)
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=2)(input_data)
		#3
		input_data = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=256,ch_in=128,k=3,padding='same',LUT=False,BINARY=BINARY,layer_index=3)(input_data)
		input_data = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)
		input_data	= MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data)
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=3)(input_data)
		#4
		input_data = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=512,ch_in=256,k=3,padding='same',LUT=False,BINARY=BINARY,layer_index=4)(input_data)
		input_data = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)

		Resid_layer_1 = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=4)
		input_data = Resid_layer_1(input_data)
		route_1 = input_data

		input_data_1, input_data_2 = tf.split(input_data, num_or_size_splits=2, axis=0)
		input_data_1=tf.squeeze(input_data_1,0)
		input_data_2=tf.squeeze(input_data_2,0)
		input_data = tf.math.add(input_data_1, input_data_2)
		
		input_data	= MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data)
		means = Resid_layer_1.means
		Resid_layer_rover = Residual_sign_recover(BINARY=BINARY,evaluate=evaluate,means=means, levels=resid_levels,trainable=False)
		input_data=Resid_layer_rover(input_data)

		
		# input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means)(input_data)
		# route_1 = input_data
		# input_data	= MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data)
		#5
		input_data = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=1024,ch_in=512,k=3,padding='same',LUT=LUT,BINARY=BINARY,layer_index=5)(input_data)
		input_data = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)
		input_data	= MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding="same")(input_data)
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=5)(input_data)
	#---------------------------------------------------------------------------------------------------------------
		#6
		conv = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=2048,ch_in=1024,k=3,padding='same',LUT=False,BINARY=BINARY,layer_index=6)(input_data)
		conv = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(conv)
		conv = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=6)(conv)
		#7
		conv = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=512,ch_in=2048,k=1,padding='same',LUT=False,BINARY=BINARY,layer_index=7)(conv)
		conv = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(conv)
		conv = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=7)(conv)
		#8
		conv_lobj_branch = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=1024,ch_in=512,k=3,padding='same',LUT=False,BINARY=BINARY,layer_index=8)(conv)
		conv_lobj_branch = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(conv_lobj_branch)
		conv_lobj_branch = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=8)(conv_lobj_branch)
		#9 Output
		conv_lbbox = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=3*(NUM_CLASS + 5),ch_in=1024,k=1,padding='same',LUT=False,BINARY=BINARY,layer_index=9)(conv_lobj_branch)
		#10
		conv = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=256,ch_in=512,k=1,padding='same',LUT=False,BINARY=BINARY,layer_index=10)(conv)
		conv = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(conv)
		Resid_layer_2 = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=9)
		conv = Resid_layer_2(conv)

		conv_1, conv_2 = tf.split(conv, num_or_size_splits=2, axis=0)
		conv_1=tf.squeeze(conv_1,0)  
		conv_2=tf.squeeze(conv_2,0)

		conv_1 = tf.image.resize(conv_1, (conv_1.shape[1] * 2, conv_1.shape[2] * 2), method='nearest')
		conv_2 = tf.image.resize(conv_2, (conv_2.shape[1] * 2, conv_2.shape[2] * 2), method='nearest')

		conv_1 = tf.expand_dims(conv_1, axis=0)
		conv_2 = tf.expand_dims(conv_2, axis=0)
		conv = tf.concat([conv_1, conv_2], axis=0)

		# conv = tf.image.resize(conv, (conv.shape[1] * 2, conv.shape[2] * 2), method='nearest')
		conv = tf.concat([conv, route_1], axis=-1)
		#11
		conv_mobj_branch = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=512,ch_in=768,k=3,padding='same',LUT=False,BINARY=BINARY,layer_index=11)(conv)	
		conv_mobj_branch = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(conv_mobj_branch)
		conv_mobj_branch = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means,layer_index=10)(conv_mobj_branch)
		#12 Output
		conv_mbbox = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=3*(NUM_CLASS + 5),ch_in=512,k=1,padding='same',LUT=False,BINARY=BINARY,layer_index=12)(conv_mobj_branch)
		
		return [conv_mbbox, conv_lbbox]

	elif dataset =="yolov3_tiny_double_channel_with_bias":
		#0
		input_data = binary_conv(evaluate=evaluate,pruning_prob=0.1,nfilters=32,ch_in=3,k=3,padding='same',input_shape=[608,608,3],first_layer=True,BINARY=BINARY)(input_layer)
		input_data = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)		
		input_data = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data)		
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means)(input_data)
		#1
		input_data = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=64,ch_in=32,k=3,padding='same',LUT=False,BINARY=BINARY)(input_data)
		input_data = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)
		input_data	= MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data)
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means)(input_data)
		#2
		input_data = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=128,ch_in=64,k=3,padding='same',LUT=False,BINARY=BINARY)(input_data)
		input_data = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)
		input_data	= MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data)
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means)(input_data)
		#3
		input_data = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=256,ch_in=128,k=3,padding='same',LUT=False,BINARY=BINARY)(input_data)
		input_data = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)
		input_data	= MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data)
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means)(input_data)
		#4
		input_data = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=512,ch_in=256,k=3,padding='same',LUT=False,BINARY=BINARY)(input_data)
		input_data = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)
		route_1 = input_data
		input_data	= MaxPooling2D(pool_size=(2, 2),strides=(2,2))(input_data)
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means)(input_data)
		#5
		input_data = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=1024,ch_in=512,k=3,padding='same',LUT=False,BINARY=BINARY)(input_data)
		input_data = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(input_data)
		input_data	= MaxPooling2D(pool_size=(2, 2),strides=(1,1),padding="same")(input_data)
		input_data = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means)(input_data)
	#---------------------------------------------------------------------------------------------------------------
		#6
		conv = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=2048,ch_in=1024,k=3,padding='same',LUT=False,BINARY=BINARY)(input_data)
		conv = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(conv)
		conv = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means)(conv)
		#7
		conv = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=512,ch_in=2048,k=1,padding='same',LUT=False,BINARY=BINARY)(conv)
		conv = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(conv)
		conv = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means)(conv)
		#8
		conv_lobj_branch = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=1024,ch_in=512,k=3,padding='same',LUT=False,BINARY=BINARY)(conv)
		conv_lobj_branch = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(conv_lobj_branch)
		conv_lobj_branch = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means)(conv_lobj_branch)
		#9 Output
		conv_lbbox = binary_conv_bias(levels=resid_levels,pruning_prob=0.1,nfilters=3*(NUM_CLASS + 5),ch_in=1024,k=1,padding='same',LUT=False,BINARY=BINARY,use_bias=True)(conv_lobj_branch)
		#10
		conv = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=256,ch_in=512,k=1,padding='same',LUT=False,BINARY=BINARY)(conv)
		conv = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(conv)

		conv = tf.image.resize(conv, (conv.shape[1] * 2, conv.shape[2] * 2), method='nearest')
		conv = tf.concat([conv, route_1], axis=-1)
		conv = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means)(conv)
		#11
		conv_mobj_branch = binary_conv(evaluate=evaluate,levels=resid_levels,pruning_prob=0.1,nfilters=512,ch_in=768,k=3,padding='same',LUT=False,BINARY=BINARY)(conv)	
		conv_mobj_branch = BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps)(conv_mobj_branch)
		conv_mobj_branch = Residual_sign(BINARY=BINARY,evaluate=evaluate,levels=resid_levels,trainable=trainable_means)(conv_mobj_branch)
		#12 Output

		conv_mbbox = binary_conv_bias(levels=resid_levels,pruning_prob=0.1,nfilters=3*(NUM_CLASS + 5),ch_in=512,k=1,padding='same',LUT=False,BINARY=BINARY,use_bias=True)(conv_mobj_branch)	
		return [conv_mbbox, conv_lbbox]

	else:
		raise("dataset should be one of the following list: [MNIST, CIFAR-10, SVHN, Imagenet].")
	return model
