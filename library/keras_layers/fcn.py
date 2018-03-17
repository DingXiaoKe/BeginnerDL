from keras.layers import Conv2D, BatchNormalization, Activation, Add
from keras.regularizers import l2
from config.FCNConfig import FCNConfig
from keras.engine.topology import Layer,InputSpec
import keras.backend as K
import tensorflow as tf
import numpy as np

cfg = FCNConfig()

def conv_block(kernel_size, filters, strides=(2,2)):
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        bn_axis = 3
        network = Conv2D(nb_filter1, 1, strides=strides, kernel_regularizer=l2(cfg.WEIGHT_DECAY))(input_tensor)
        network = BatchNormalization(axis=bn_axis, momentum=cfg.BATCH_MOMENTUM)(network)
        network = Activation('relu')(network)

        network = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', kernel_regularizer=l2(cfg.WEIGHT_DECAY))(network)
        network = BatchNormalization(axis=bn_axis, momentum=cfg.BATCH_MOMENTUM)(network)
        network = Activation('relu')(network)

        network = Conv2D(nb_filter3, (1,1), kernel_regularizer=l2(cfg.WEIGHT_DECAY))(network)
        network = BatchNormalization(axis=bn_axis, momentum=cfg.BATCH_MOMENTUM)(network)

        shortcut = Conv2D(nb_filter3, (1,1), strides=strides, kernel_regularizer=l2(cfg.WEIGHT_DECAY))(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, momentum=cfg.BATCH_MOMENTUM)(shortcut)

        network = Add()([network, shortcut])
        network = Activation('relu')(network)

        return network
    return f

def identity_block(kernel_size, filters):
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        bn_axis = 3
        x = Conv2D(nb_filter1, (1, 1), kernel_regularizer=l2(cfg.WEIGHT_DECAY))(input_tensor)
        x = BatchNormalization(axis=bn_axis, momentum=cfg.BATCH_MOMENTUM)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size),
                   padding='same',kernel_regularizer=l2(cfg.WEIGHT_DECAY))(x)
        x = BatchNormalization(axis=bn_axis, momentum=cfg.BATCH_MOMENTUM)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1),kernel_regularizer=l2(cfg.WEIGHT_DECAY))(x)
        x = BatchNormalization(axis=bn_axis, momentum=cfg.BATCH_MOMENTUM)(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x
    return f

def atrous_conv_block(kernel_size, filters, strides=(1, 1), atrous_rate=(2, 2)):
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        bn_axis = 3

        x = Conv2D(nb_filter1, (1, 1), strides=strides, kernel_regularizer=l2(cfg.WEIGHT_DECAY))(input_tensor)
        x = BatchNormalization(axis=bn_axis, momentum=cfg.BATCH_MOMENTUM)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', dilation_rate=atrous_rate,kernel_regularizer=l2(cfg.WEIGHT_DECAY))(x)
        x = BatchNormalization(axis=bn_axis, momentum=cfg.BATCH_MOMENTUM)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1),kernel_regularizer=l2(cfg.WEIGHT_DECAY))(x)
        x = BatchNormalization(axis=bn_axis, momentum=cfg.BATCH_MOMENTUM)(x)

        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,kernel_regularizer=l2(cfg.WEIGHT_DECAY))(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis,momentum=cfg.BATCH_MOMENTUM)(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    return f

def atrous_identity_block(kernel_size, filters, atrous_rate=(2, 2)):
    def f(input_tensor):
        nb_filter1, nb_filter2, nb_filter3 = filters
        bn_axis = 3

        x = Conv2D(nb_filter1, (1, 1), kernel_regularizer=l2(cfg.WEIGHT_DECAY))(input_tensor)
        x = BatchNormalization(axis=bn_axis, momentum=cfg.BATCH_MOMENTUM)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate,padding='same', kernel_regularizer=l2(cfg.WEIGHT_DECAY))(x)
        x = BatchNormalization(axis=bn_axis,momentum=cfg.BATCH_MOMENTUM)(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), kernel_regularizer=l2(cfg.WEIGHT_DECAY))(x)
        x = BatchNormalization(axis=bn_axis, momentum=cfg.BATCH_MOMENTUM)(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x
    return f

class BilinearUpSampling2D(Layer):
    def __init__(self, size=(1, 1), target_size=None, data_format='default', **kwargs):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpSampling2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.data_format == 'channels_last':
            width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1], data_format=self.data_format)
        else:
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1], data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def resize_images_bilinear(X, height_factor=1, width_factor=1, target_height=None, target_width=None, data_format='default'):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if data_format == 'default':
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = K.permute_dimensions(X, [0, 2, 3, 1])
        X = tf.image.resize_bilinear(X, new_shape)
        X = K.permute_dimensions(X, [0, 3, 1, 2])
        if target_height and target_width:
            X.set_shape((None, None, target_height, target_width))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor))
        return X
    elif data_format == 'channels_last':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = tf.image.resize_bilinear(X, new_shape)
        if target_height and target_width:
            X.set_shape((None, target_height, target_width, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, None))
        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)