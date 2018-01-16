from keras.layers import Conv2D, BatchNormalization, Concatenate, GlobalAveragePooling2D, Dense, Reshape, \
    AveragePooling2D, Activation, ZeroPadding2D
import numpy as np

def transform_layer(x, training, stride, depth = 8):
    x = Conv2D(filters=depth, kernel_size=(1,1), strides=(1,1), padding="SAME")(x)
    x = BatchNormalization(trainable=training)(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=depth, kernel_size=(3,3), strides=stride, padding="SAME")(x)
    x = BatchNormalization(trainable=training)(x)
    x = Activation('relu')(x)

    return x

def transition_layer(x, training, out_dim):
    x = Conv2D(filters=out_dim, kernel_size=(1,1), strides=(1,1), padding="SAME")(x)
    x = BatchNormalization(trainable=training)(x)

    return x

def split_layer(input_x, training,  stride, cardinality=8):
    layers_list = list()

    for i in range(cardinality):
        splits = transform_layer(input_x, training=training, stride=stride)
        layers_list.append(splits)

    return Concatenate()(layers_list)

def squeeze_excitation_layer(input_x, out_dim, ratio):
    squeeze = GlobalAveragePooling2D()(input_x)
    excitation = Dense(units=out_dim / ratio)(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = Reshape([-1,1,1,out_dim])(excitation)
    scale = input_x * excitation

    return scale

def residual_layer(input_x, out_dim, training, res_block=3, reduction_ratio=4):
    for i in range(res_block):
        input_dim = int(np.shape(input_x)[-1])

        if input_dim * 2 == out_dim:
            flag = True
            stride = 2
            channel = input_dim // 2
        else:
            flag = False
            stride = 1

        x = split_layer(input_x, training=training, stride=stride)
        x = transition_layer(x, training=training, out_dim=out_dim)
        x = squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio)

        if flag is True :
            pad_input_x = AveragePooling2D(padding="SAME")(input_x)
            # pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]])
            pad_input_x = ZeroPadding2D(padding=((0,0), (0, channel)))(pad_input_x)
        else :
            pad_input_x = input_x

        input_x = Activation('relu')(x + pad_input_x)

    return x