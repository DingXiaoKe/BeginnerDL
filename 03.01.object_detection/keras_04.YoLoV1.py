from keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
import keras.backend as K
from keras.utils import multi_gpu_model
import tensorflow as tf
import math
from keras.models import Model
from keras_losses.yolov1 import loss_layer
from keras.optimizers import SGD
from keras_datareaders.yolov1_voc_reader import pascal_voc
from utils.progressbar.keras.ProgressBarCallback import ProgressBarCallback
from keras_config.yoloV1Config import YoloV1Config
from keras.preprocessing.image import ImageDataGenerator

cfg = YoloV1Config()

def newModel(images):
    ALPHA = cfg.LEAKEY_ALPHA
    network = ZeroPadding2D((3,3))(images)
    network = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='valid', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = MaxPooling2D(pool_size=(2,2), padding='same')(network)

    network = Conv2D(filters=192, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = MaxPooling2D(pool_size=(2,2), padding='same')(network)

    network = Conv2D(filters=128, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=256, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=256, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=512, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = MaxPooling2D(pool_size=(2,2), padding='same')(network)

    network = Conv2D(filters=256, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=512, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=256, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=512, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=256, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=512, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=256, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=512, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=512, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=1024, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = MaxPooling2D(pool_size=(2,2), padding='same')(network)

    network = Conv2D(filters=512,  kernel_size=(1,1), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=1024, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=512, kernel_size=(1,1), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=1024, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=1024, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = ZeroPadding2D(padding=(1,1))(network)
    network = Conv2D(filters=1024, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=1024, kernel_size=(3,3), padding='same', kernel_initializer='he_normal',kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Conv2D(filters=1024, kernel_size=(3,3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    # network = K.tf.transpose(network, perm=[0,3,1,2])
    # network = tf.transpose(network, perm=[0,3,1,2])
    network = Lambda(lambda x : K.tf.transpose(x, perm=[0,3,1,2]))(network)
    network = Flatten()(network)
    network = Dense(units=512, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Dense(units=4096, kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = Dropout(rate=cfg.KEEP_PROB)(network)
    network = Dense(units=cfg.NUM_OUTPUTS,kernel_initializer='he_normal', kernel_regularizer=l2(0.0005))(network)

    model = Model(inputs=images, outputs=network)
    return model

def lr_schedule(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate


session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
K.set_session(session)
image_input = Input(shape=(cfg.IMAGE_SIZE,cfg.IMAGE_SIZE,cfg.IMAGE_CHANNEL))
model = newModel(image_input)
if cfg.GPU_NUM >= 2:
    model = multi_gpu_model(model, gpus=cfg.GPU_NUM)

model.compile(loss=loss_layer,
              optimizer=SGD(lr=0.001, momentum=0.9, nesterov=True),
              metrics=['acc'])

train_image_data_generator = ImageDataGenerator(
    horizontal_flip=True,
)

reader = pascal_voc(cfg.BATCH_SIZE,train_image_data_generator, cfg.DATAPATH)
reader.prepare(model="train")
probar = ProgressBarCallback()
model.fit_generator(generator=reader,steps_per_epoch=5011/cfg.BATCH_SIZE,
                    epochs=cfg.EPOCH_NUM, verbose=0,callbacks=[probar])
model.save_weights("yolo_v1.h5")
print("Saved model to disk")

