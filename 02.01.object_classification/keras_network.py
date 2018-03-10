import tensorflow as tf
import math
import scipy

import keras.backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy
from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from lib.utils.progressbar.keras.ProgressBarCallback import ProgressBar
from lib.datareader.DataReaderForClassification import DataReader
from lib.config.cifarConfig import Cifar10Config
from lib.models.keras.cifar import AlexNet,Vgg19,SENet,DenseNet,ResNet

cfg = Cifar10Config()
MODEL = "alexnet"

MODEL_LIST = [
    {
        "name" : "alexnet",
        "model" : AlexNet(cfg)
    },
    {
        "name" : "Vgg19",
        "model" : Vgg19(cfg, zoom=2)
    },
    {
        "name" : "SENet",
        "model" : SENet(cfg)
    },
    {
        "name" : "ResNet",
        "model" : ResNet(cfg, zoom=7)
    },
    {
        "name" : "DenseNet",
        "model" : DenseNet(cfg, zoom=1)
    }
]

GPU_NUM = 1
EPOCH_NUM = 100

def lr_schedule(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
K.set_session(session)
model = MODEL_LIST["name"==MODEL]["model"]
zoom = model.zoom
print(zoom)
model = model.network()

if GPU_NUM >= 2:
    model = multi_gpu_model(model, gpus=GPU_NUM)


model.compile(loss=categorical_crossentropy,
              optimizer=SGD(lr=0.0, momentum=0.9, nesterov=True),
              metrics=['acc'])

train_dataGen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_dataGen.flow_from_directory(directory="../data/cifar10/train",
                                                    target_size=(cfg.IMAGE_SIZE*zoom,cfg.IMAGE_SIZE*zoom), batch_size=cfg.BATCH_SIZE,
                                                    class_mode='categorical')
probar = ProgressBar()
es = EarlyStopping(monitor='val_acc', patience=EPOCH_NUM)
checkpoint = ModelCheckpoint(filepath="cifar10_{%s}.h5" % MODEL, save_best_only=True, save_weights_only=True)
lrate = LearningRateScheduler(lr_schedule)

reader = DataReader()
x_test, y_test = reader.readData(phrase="test")
y_test = to_categorical(y_test, num_classes=cfg.NUM_OUTPUTS)
x_test = scipy.misc.imresize(x_test, cfg.IMAGE_SIZE * zoom, interp='cubic')

model.fit_generator(generator=train_generator, steps_per_epoch=cfg.TRAIN_RECORDS/cfg.BATCH_SIZE,
                    epochs=EPOCH_NUM, verbose=0,
                    validation_data=(x_test, y_test),validation_steps=cfg.TEST_RECORDS/cfg.BATCH_SIZE,
                    callbacks=[probar,es,checkpoint, lrate])

# scores = model.evaluate(x_test, y_test, verbose=1, steps=EPOCH_NUM)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])

