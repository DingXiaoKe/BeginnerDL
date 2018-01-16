import numpy as np
from keras.layers import (Conv2D, Input, Lambda, Concatenate, MaxPooling2D)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

image_input = Input(shape=(416,416,3),name='input_1')
#先将ｗｅｉｇｈｔ读入这个变量，且扔掉开头四个数字
weights_file = open('../weights/yolo_v2.weights','rb')
count = 16
# 挨着增加层数且直接把ｗｅｉｇｈｔ附上
#不使用ｃｏｕｎｔ
weights_file.read(16)

bias = np.ndarray(shape=(32,),dtype='float32',buffer=weights_file.read(32*4))
bn_list = np.ndarray(shape=(3, 32),dtype='float32',buffer=weights_file.read(32*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(32,3,3,3),dtype='float32',buffer=weights_file.read(32*3*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(32,3,padding="same", weights=[weights], use_bias=False, kernel_regularizer=l2(0.0005), name='conv2d_1')(image_input)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_1')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_1')(tmp)
count += (32 + 32*3 + 3*3*3*32)*4
print ("file read to",count)
tmp = MaxPooling2D(pool_size=2,strides=2,padding='same',name='maxpooling2d_1')(tmp)

bias = np.ndarray(shape=(64,),dtype='float32',buffer=weights_file.read(64*4))
bn_list = np.ndarray(shape=(3, 64),dtype='float32',buffer=weights_file.read(64*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(64,32,3,3),dtype='float32',buffer=weights_file.read(64*32*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(64,3,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_2')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_2')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_2')(tmp)
count += (64 + 64*3 + 3*3*64*32)*4
print ("file read to",count)
tmp = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name='maxpooling2d_2')(tmp)

bias = np.ndarray(shape=(128,),dtype='float32',buffer=weights_file.read(128*4))
bn_list = np.ndarray(shape=(3, 128),dtype='float32',buffer=weights_file.read(128*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(128,64,3,3),dtype='float32',buffer=weights_file.read(128*64*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(128,3,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_3')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_3')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_3')(tmp)
count += (128 + 128*3 + 3*3*128*64)*4
print ("file read to",count)

bias = np.ndarray(shape=(64,),dtype='float32',buffer=weights_file.read(64*4))
bn_list = np.ndarray(shape=(3, 64),dtype='float32',buffer=weights_file.read(64*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(64,128,1,1),dtype='float32',buffer=weights_file.read(64*128*1*1*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(64,1,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_4')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_4')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_4')(tmp)
count += (64 + 64*3 + 1*1*64*128)*4
print ("file read to",count)

bias = np.ndarray(shape=(128,),dtype='float32',buffer=weights_file.read(128*4))
bn_list = np.ndarray(shape=(3, 128),dtype='float32',buffer=weights_file.read(128*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(128,64,3,3),dtype='float32',buffer=weights_file.read(128*64*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(128,3,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_5')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_5')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_5')(tmp)
count += (128 + 128*3 + 3*3*128*64)*4
print ("file read to",count)
tmp = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name='maxpooling2d_3')(tmp)

bias = np.ndarray(shape=(256,),dtype='float32',buffer=weights_file.read(256*4))
bn_list = np.ndarray(shape=(3, 256),dtype='float32',buffer=weights_file.read(256*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(256,128,3,3),dtype='float32',buffer=weights_file.read(256*128*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(256,3,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_6')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_6')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_6')(tmp)
count += (256 + 256*3 + 3*3*256*128)*4
print ("file read to",count)

bias = np.ndarray(shape=(128,),dtype='float32',buffer=weights_file.read(128*4))
bn_list = np.ndarray(shape=(3, 128),dtype='float32',buffer=weights_file.read(128*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(128,256,1,1),dtype='float32',buffer=weights_file.read(128*256*1*1*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(128,1,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_7')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_7')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_7')(tmp)
count += (128 + 128*3 + 1*1*128*256)*4
print ("file read to",count)

bias = np.ndarray(shape=(256,),dtype='float32',buffer=weights_file.read(256*4))
bn_list = np.ndarray(shape=(3, 256),dtype='float32',buffer=weights_file.read(256*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(256,128,3,3),dtype='float32',buffer=weights_file.read(256*128*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(256,3,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_8')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_8')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_8')(tmp)
count += (256 + 256*3 + 3*3*256*128)*4
print ("file read to",count)
tmp = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name='maxpooling2d_4')(tmp)

bias = np.ndarray(shape=(512,),dtype='float32',buffer=weights_file.read(512*4))
bn_list = np.ndarray(shape=(3, 512),dtype='float32',buffer=weights_file.read(512*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(512,256,3,3),dtype='float32',buffer=weights_file.read(512*256*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(512,3,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_9')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_9')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_9')(tmp)
count += (512 + 512*3 + 3*3*512*256)*4
print ("file read to",count)

bias = np.ndarray(shape=(256,),dtype='float32',buffer=weights_file.read(256*4))
bn_list = np.ndarray(shape=(3, 256),dtype='float32',buffer=weights_file.read(256*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(256,512,1,1),dtype='float32',buffer=weights_file.read(256*512*1*1*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(256,1,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_10')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_10')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_10')(tmp)
count += (256 + 256*3 + 1*1*256*512)*4
print ("file read to",count)


bias = np.ndarray(shape=(512,),dtype='float32',buffer=weights_file.read(512*4))
bn_list = np.ndarray(shape=(3, 512),dtype='float32',buffer=weights_file.read(512*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(512,256,3,3),dtype='float32',buffer=weights_file.read(512*256*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(512,3,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_11')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_11')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_11')(tmp)
count += (512 + 512*3 + 3*3*512*256)*4
print ("file read to",count)

bias = np.ndarray(shape=(256,),dtype='float32',buffer=weights_file.read(256*4))
bn_list = np.ndarray(shape=(3, 256),dtype='float32',buffer=weights_file.read(256*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(256,512,1,1),dtype='float32',buffer=weights_file.read(256*512*1*1*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(256,1,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_12')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_12')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_12')(tmp)
count += (256 + 256*3 + 1*1*256*512)*4
print ("file read to",count)

bias = np.ndarray(shape=(512,),dtype='float32',buffer=weights_file.read(512*4))
bn_list = np.ndarray(shape=(3, 512),dtype='float32',buffer=weights_file.read(512*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(512,256,3,3),dtype='float32',buffer=weights_file.read(512*256*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(512,3,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_13')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_13')(tmp)
image_tmp_output = LeakyReLU(alpha=0.1,name='leakyrelu_13')(tmp)
count += (512 + 512*3 + 3*3*512*256)*4
print ("file read to",count)
tmp = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name='maxpooling2d_5')(image_tmp_output)

bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(1024,512,3,3),dtype='float32',buffer=weights_file.read(1024*512*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(1024,3,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_14')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_14')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_14')(tmp)
count += (1024 + 1024*3 + 3*3*1024*512)*4
print ("file read to",count)

bias = np.ndarray(shape=(512,),dtype='float32',buffer=weights_file.read(512*4))
bn_list = np.ndarray(shape=(3, 512),dtype='float32',buffer=weights_file.read(512*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(512,1024,1,1),dtype='float32',buffer=weights_file.read(512*1024*1*1*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(512,1,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_15')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_15')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_15')(tmp)
count += (512 + 512*3 + 1*1*512*1024)*4
print ("file read to",count)

bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(1024,512,3,3),dtype='float32',buffer=weights_file.read(1024*512*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(1024,3,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_16')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_16')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_16')(tmp)
count += (1024 + 1024*3 + 3*3*1024*512)*4
print ("file read to",count)

bias = np.ndarray(shape=(512,),dtype='float32',buffer=weights_file.read(512*4))
bn_list = np.ndarray(shape=(3, 512),dtype='float32',buffer=weights_file.read(512*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(512,1024,1,1),dtype='float32',buffer=weights_file.read(512*1024*1*1*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Conv2D(512,1,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),
                    name='conv2d_17')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_17')(tmp)

#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_17')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (512 + 512*3 + 1*1*512*1024)*4
print ("file read to",count)
"""
convolutional_17
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(1024,512,3,3),dtype='float32',buffer=weights_file.read(1024*512*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Conv2D(1024,3,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),
                    name='conv2d_18')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_18')(tmp)

#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_18')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (1024 + 1024*3 + 3*3*1024*512)*4
print ("file read to",count)
"""
convolutional_18
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(1024,1024,3,3),dtype='float32',buffer=weights_file.read(1024*1024*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp =Conv2D(1024,3,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),
                    name='conv2d_19')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_19')(tmp)

#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_19')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (1024 + 1024*3 + 3*3*1024*1024)*4
print ("file read to",count)
"""
convolutional_19
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(1024,1024,3,3),dtype='float32',buffer=weights_file.read(1024*1024*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp = Conv2D(1024,3,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),
                    name='conv2d_20')(tmp)

#batchnormalization
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_20')(tmp)

#activation
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_20')(tmp)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (1024 + 1024*3 + 3*3*1024*1024)*4
print ("file read to",count)
"""
分叉处
convolutional_20
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky
"""
#read weights from yolo.weights
#weights_file.seek(count)
bias = np.ndarray(shape=(64,),dtype='float32',buffer=weights_file.read(64*4))
bn_list = np.ndarray(shape=(3, 64),dtype='float32',buffer=weights_file.read(64*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(64,512,1,1),dtype='float32',buffer=weights_file.read(64*512*1*1*4))
weights = np.transpose(weights,(2,3,1,0))

#read for convolution
tmp2 = Conv2D(64,1,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),
                     name='conv2d_21')(image_tmp_output)

#batchnormalization
tmp2 = BatchNormalization(weights=bn_weights,name='batch_normalization_21')(tmp2)

#activation
tmp2 = LeakyReLU(alpha=0.1,name='leakyrelu_21')(tmp2)

#help to go back. 因为file.seek(count)是默认从０开始算起到的位置
count += (64 + 64*3 + 1*1*64*512)*4
print ("file read to",count)
#需要和之前的image_tmp_output融合了
def fun(x):
    import tensorflow as tf
    return tf.space_to_depth(x, block_size=2)
tmp2 = Lambda(fun,output_shape=(13,13,256),name='space_to_depth_2')(tmp2)
tmp = Concatenate()([tmp2,tmp])

bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
weights = np.ndarray(shape=(1024,1280,3,3),dtype='float32',buffer=weights_file.read(1024*1280*3*3*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(1024,3,padding='same',weights=[weights],use_bias=False,kernel_regularizer=l2(0.0005),name='conv2d_22')(tmp)
tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_22')(tmp)
tmp = LeakyReLU(alpha=0.1,name='leakyrelu_22')(tmp)
count += (1024 + 1024*3 + 3*3*1024*1280)*4
print ("file read to",count)

bias = np.ndarray(shape=(425,),dtype='float32',buffer=weights_file.read(425*4))
weights = np.ndarray(shape=(425,1024,1,1),dtype='float32',buffer=weights_file.read(425*1024*1*1*4))
weights = np.transpose(weights,(2,3,1,0))

tmp = Conv2D(425,1,padding='same',weights=[weights, bias],use_bias=True,kernel_regularizer=l2(0.0005),name='conv2d_23')(tmp)

count += (1*1*1024*425)*4
print ("file read to",count)
model = Model(inputs=image_input, outputs=tmp)
model.save('../weights/model_yolov2_416.h5')
weights_file.close()