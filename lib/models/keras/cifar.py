from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization,Activation,\
    Flatten,Dense,Dropout,GlobalAveragePooling2D,AveragePooling2D,ZeroPadding2D,Concatenate, \
    Reshape
from keras.applications import VGG19, ResNet50, DenseNet121

from keras.models import Model
from lib.config.cifarConfig import Cifar10Config
import numpy as np

class AlexNet(object):
    def __init__(self, config=Cifar10Config(), zoom=1):
        self.cfg = config
        self.zoom = zoom

    def network(self):
        image_input = Input(shape=(self.cfg.IMAGE_SIZE, self.cfg.IMAGE_SIZE, self.cfg.IMAGE_CHANNEL))
        network = Conv2D(filters=96, kernel_size=(5,5), kernel_initializer="he_normal", padding="same")(image_input)
        network = BatchNormalization()(network)
        network = Activation('relu')(network)
        network = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(network)

        network = Conv2D(filters=256, kernel_size=(5,5), padding="same", kernel_initializer="he_normal")(network)
        network = BatchNormalization()(network)
        network = Activation('relu')(network)
        network = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(network)

        network = Conv2D(filters=384, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
        network = BatchNormalization()(network)
        network = Activation('relu')(network)

        network = Conv2D(filters=384, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
        network = BatchNormalization()(network)
        network = Activation('relu')(network)

        network = Conv2D(filters=256, kernel_size=(3,3), padding="same",kernel_initializer="he_normal")(network)
        network = BatchNormalization()(network)
        network = Activation('relu')(network)
        network = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(network)

        network = Flatten()(network)
        network = Dense(units=4096)(network)
        network = BatchNormalization()(network)
        network = Activation('relu')(network)
        network = Dropout(rate=0.5)(network)

        network = Dense(units=4096)(network)
        network = BatchNormalization()(network)
        network = Activation('relu')(network)
        network = Dropout(rate=0.5)(network)

        network = Dense(units=self.cfg.NUM_OUTPUTS, activation="softmax")(network)

        model = Model(inputs=image_input, outputs=network)
        return model

class SENet(object):
    def __init__(self, config=Cifar10Config(), zoom=1):
        self.cfg = config
        self.zoom = zoom

    def network(self):
        image_input = Input(shape=(self.cfg.IMAGE_SIZE, self.cfg.IMAGE_SIZE, self.cfg.IMAGE_CHANNEL))
        network = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="SAME")(image_input)
        network = BatchNormalization(trainable=True)(network)
        network = Activation('relu')(network)
        network = self._residual_layer(network, out_dim=64, training=True)
        network = self._residual_layer(network, out_dim=128, training=True)
        network = self._residual_layer(network, out_dim=256, training=True)
        network = GlobalAveragePooling2D()(network)
        network = Flatten()(network)
        network = Dense(units=self.cfg.NUM_OUTPUTS, activation="softmax")(network)

        model = Model(inputs=image_input, outputs=network)
        return model

    def _residual_layer(self, input_x, out_dim, training, res_block=3, reduction_ratio=4):
        for i in range(res_block):
            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1

            x = self._split_layer(input_x, training=training, stride=stride)
            x = self._transition_layer(x, training=training, out_dim=out_dim)
            x = self._squeeze_excitation_layer(x, out_dim=out_dim, ratio=reduction_ratio)

            if flag is True :
                pad_input_x = AveragePooling2D(padding="SAME")(input_x)
                # pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]])
                pad_input_x = ZeroPadding2D(padding=((0,0), (0, channel)))(pad_input_x)
            else :
                pad_input_x = input_x

            input_x = Activation('relu')(x + pad_input_x)
        return x

    def _split_layer(self,input_x, training,  stride, cardinality=8):
        layers_list = list()

        for i in range(cardinality):
            splits = self._transform_layer(input_x, training=training, stride=stride)
            layers_list.append(splits)

        return Concatenate()(layers_list)

    def _transition_layer(self, x, training, out_dim):
        x = Conv2D(filters=out_dim, kernel_size=(1,1), strides=(1,1), padding="SAME")(x)
        x = BatchNormalization(trainable=training)(x)

        return x

    def _squeeze_excitation_layer(self,input_x, out_dim, ratio):
        squeeze = GlobalAveragePooling2D()(input_x)
        excitation = Dense(units=out_dim / ratio)(squeeze)
        excitation = Activation('relu')(excitation)
        excitation = Dense(units=out_dim)(excitation)
        excitation = Activation('sigmoid')(excitation)
        excitation = Reshape([-1,1,1,out_dim])(excitation)
        scale = input_x * excitation

        return scale

    def _transform_layer(self, x, training, stride, depth = 8):
        x = Conv2D(filters=depth, kernel_size=(1,1), strides=(1,1), padding="SAME")(x)
        x = BatchNormalization(trainable=training)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters=depth, kernel_size=(3,3), strides=stride, padding="SAME")(x)
        x = BatchNormalization(trainable=training)(x)
        x = Activation('relu')(x)

        return x

class Vgg19(object):
    def __init__(self,config=Cifar10Config(), zoom=2):
        self.cfg = config
        self.zoom = zoom

    def network(self):
        return VGG19(include_top=True, weights=None,
                     input_shape=(self.cfg.IMAGE_SIZE * self.zoom,self.cfg.IMAGE_SIZE * self.zoom, self.cfg.IMAGE_CHANNEL),
                     classes=self.cfg.NUM_OUTPUTS)

class ResNet(object):
    def __init__(self,config=Cifar10Config(), zoom=2):
        self.cfg = config
        self.zoom = zoom

    def network(self):
        return ResNet50(include_top=True, weights=None,
                        input_shape=(self.cfg.IMAGE_SIZE * self.zoom, self.cfg.IMAGE_SIZE * self.zoom, self.cfg.IMAGE_CHANNEL),
                        classes=self.cfg.NUM_OUTPUTS)
class DenseNet(object):
    def __init__(self,config=Cifar10Config(), zoom=1):
        self.cfg = config
        self.zoom = zoom

    def network(self):
        return DenseNet121(include_top=True, weights=None,
                        input_shape=(self.cfg.IMAGE_SIZE * self.zoom, self.cfg.IMAGE_SIZE * self.zoom, self.cfg.IMAGE_CHANNEL),
                        classes=self.cfg.NUM_OUTPUTS)