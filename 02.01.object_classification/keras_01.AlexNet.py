import tensorflow as tf
import math
import os

import keras.backend as K
from keras import losses as Klosses
from keras import optimizers as Koptimizers
from keras import utils as Kutils
from keras.preprocessing import image as Kimage
from keras import callbacks as Kcallbacks

from lib.utils.progressbar.keras.ProgressBarCallback import ProgressBar
from lib.datareader.DataReaderForClassification import DataReader
from lib.config.cifarConfig import Cifar10Config
from lib.models.keras.cifar import alexnet

cfg = Cifar10Config()

GPU_NUM = cfg.GPU_NUM
EPOCH_NUM = cfg.EPOCH_NUM

def lr_schedule(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
K.set_session(session)
model = alexnet()
if GPU_NUM >= 2:
    model = Kutils.multi_gpu_model(model, gpus=GPU_NUM)

model.compile(loss=Klosses.categorical_crossentropy,
              optimizer=Koptimizers.SGD(lr=0.0, momentum=0.9, nesterov=True),
              metrics=['acc'])

train_dataGen = Kimage.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_dataGen.flow_from_directory(directory=os.path.join(cfg.DATAPATH, "cifar10", "train"),
                                                    target_size=(cfg.IMAGE_SIZE,cfg.IMAGE_SIZE), batch_size=cfg.BATCH_SIZE,
                                  class_mode='categorical')
probar = ProgressBar()
es = Kcallbacks.EarlyStopping(monitor='val_acc', patience=EPOCH_NUM)
checkpoint = Kcallbacks.ModelCheckpoint(filepath="cifar10_alexnet.h5", save_best_only=True, save_weights_only=True)
lrate = Kcallbacks.LearningRateScheduler(lr_schedule)

reader = DataReader(dataPath=os.path.join(cfg.DATAPATH, "cifar10"))
x_test, y_test = reader.readData(phrase="test")
y_test = Kutils.to_categorical(y_test, num_classes=cfg.NUM_OUTPUTS)

model.fit_generator(generator=train_generator, steps_per_epoch=50000/cfg.BATCH_SIZE,
                    epochs=EPOCH_NUM, verbose=0,
                    validation_data=(x_test, y_test),validation_steps=10000/cfg.BATCH_SIZE,
                    callbacks=[probar,es,checkpoint, lrate])

