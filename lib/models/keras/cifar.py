from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization,Activation,\
    Flatten,Dense,Dropout,Add,AveragePooling2D,GlobalAveragePooling2D
from keras.models import Model

from lib.config.cifarConfig import Cifar10Config

# from keras_layers.resnet import resnet_block
# from keras_layers.senet import residual_layer
# from keras_layers.densenet import dense_block,transition_block
# from keras.regularizers import l2
cfg = Cifar10Config()

def alexnet():
    image_input = Input(shape=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_CHANNEL))
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

    network = Dense(units=cfg.NUM_OUTPUTS, activation="softmax")(network)

    model = Model(inputs=image_input, outputs=network)
    return model

# def vgg19():
#     image_input = Input(shape=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_CHANNEL))
#     network = Conv2D(filters=64, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(image_input)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = Conv2D(filters=64, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(network)
#
#     network = Conv2D(filters=128, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = Conv2D(filters=128, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(network)
#
#     network = Conv2D(filters=256, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = Conv2D(filters=256, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = Conv2D(filters=256, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = Conv2D(filters=256, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(network)
#
#     network = Conv2D(filters=512, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = Conv2D(filters=512, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = Conv2D(filters=512, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = Conv2D(filters=512, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(network)
#
#     network = Conv2D(filters=512, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = Conv2D(filters=512, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = Conv2D(filters=512, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = Conv2D(filters=512, kernel_size=(3,3), padding="same", kernel_initializer="he_normal")(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#
#     network = Flatten()(network)
#     network = Dense(units=4096)(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = Dropout(rate=0.5)(network)
#
#     network = Dense(units=4096)(network)
#     network = BatchNormalization()(network)
#     network = Activation('relu')(network)
#     network = Dropout(rate=0.5)(network)
#
#     network = Dense(units=cfg.NUM_OUTPUTS)(network)
#
#     model = Model(inputs=image_input, outputs=network)
#     return model
#
# def resnet_V1(depth):
#     inputs = Input(shape=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_CHANNEL))
#
#     if (depth - 2) % 6 != 0:
#         raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
#
#     num_filters = 16
#     num_sub_blocks = int((depth - 2) / 6)
#
#     x = resnet_block(inputs=inputs)
#     # Instantiate convolutional base (stack of blocks).
#     for i in range(3):
#         for j in range(num_sub_blocks):
#             strides = 1
#             is_first_layer_but_not_first_block = j == 0 and i > 0
#             if is_first_layer_but_not_first_block:
#                 strides = 2
#             y = resnet_block(inputs=x,
#                              num_filters=num_filters,
#                              strides=strides)
#             y = resnet_block(inputs=y,
#                              num_filters=num_filters,
#                              activation=None)
#             if is_first_layer_but_not_first_block:
#                 x = resnet_block(inputs=x,
#                                  num_filters=num_filters,
#                                  kernel_size=1,
#                                  strides=strides,
#                                  activation=None)
#             x = Add()([x, y])
#             x = Activation('relu')(x)
#         num_filters = 2 * num_filters
#
#     # Add classifier on top.
#     # v1 does not use BN after last shortcut connection-ReLU
#     x = AveragePooling2D(pool_size=8)(x)
#     y = Flatten()(x)
#     outputs = Dense(cfg.NUM_OUTPUTS,
#                     activation='softmax',
#                     kernel_initializer='he_normal')(y)
#
#     # Instantiate model.
#     model = Model(inputs=inputs, outputs=outputs)
#     return model
#
# def resnet_V2(depth):
#     inputs = Input(shape=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_CHANNEL))
#     if (depth - 2) % 6 != 0:
#         raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
#
#     num_filters = 16
#     num_sub_blocks = int((depth - 2) / 6)
#
#     x = resnet_block(inputs=inputs)
#     # Instantiate convolutional base (stack of blocks).
#     for i in range(3):
#         for j in range(num_sub_blocks):
#             strides = 1
#             is_first_layer_but_not_first_block = j == 0 and i > 0
#             if is_first_layer_but_not_first_block:
#                 strides = 2
#             y = resnet_block(inputs=x,
#                              num_filters=num_filters,
#                              strides=strides)
#             y = resnet_block(inputs=y,
#                              num_filters=num_filters,
#                              activation=None)
#             if is_first_layer_but_not_first_block:
#                 x = resnet_block(inputs=x,
#                                  num_filters=num_filters,
#                                  kernel_size=1,
#                                  strides=strides,
#                                  activation=None)
#             x = Add()([x, y])
#             x = Activation('relu')(x)
#         num_filters = 2 * num_filters
#
#     # Add classifier on top.
#     # v1 does not use BN after last shortcut connection-ReLU
#     x = AveragePooling2D(pool_size=8)(x)
#     y = Flatten()(x)
#     outputs = Dense(cfg.NUM_OUTPUTS,
#                     activation='softmax',
#                     kernel_initializer='he_normal')(y)
#
#     # Instantiate model.
#     model = Model(inputs=inputs, outputs=outputs)
#     return model
#
# def densenet(depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
#                    weight_decay=1E-4, verbose=True):
#     model_input = Input(shape=(cfg.IMAGE_SIZE,cfg.IMAGE_SIZE,cfg.IMAGE_CHANNEL))
#
#     concat_axis = 3
#     assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"
#     nb_layers = int((depth - 4) / 3)
#
#     # Initial convolution
#     x = Conv2D(filters=nb_filter, kernel_size=(3, 3), kernel_initializer="he_uniform", padding="same", name="initial_conv2D", use_bias=False,
#                       kernel_regularizer=l2(weight_decay))(model_input)
#     x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
#                            beta_regularizer=l2(weight_decay))(x)
#     x = Activation('relu')(x)
#
#     for block_idx in range(nb_dense_block - 1):
#         x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
#                                    weight_decay=weight_decay)
#         x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)
#
#     x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
#                                weight_decay=weight_decay)
#     x = Activation('relu')(x)
#     x = GlobalAveragePooling2D()(x)
#     x = Dense(cfg.NUM_OUTPUTS, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)
#
#     densenet = Model(inputs=model_input, outputs=x)
#     return densenet
#
# def senet():
#     image_input = Input(shape=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_CHANNEL))
#     network = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="SAME")(image_input)
#     network = BatchNormalization(trainable=True)(network)
#     network = Activation('relu')(network)
#     network = residual_layer(network, out_dim=64, training=True)
#     network = residual_layer(network, out_dim=128, training=True)
#     network = residual_layer(network, out_dim=256, training=True)
#     network = GlobalAveragePooling2D()(network)
#     network = Flatten()(network)
#     network = Dense(units=cfg.NUM_OUTPUTS, activation="softmax")(network)
#
#     model = Model(inputs=image_input, outputs=network)
#     return model