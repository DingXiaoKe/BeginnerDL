import tensorflow as tf
from utils.progressbar.keras.ProgressBarCallback import ProgressBarCallback
from keras.preprocessing.image import ImageDataGenerator
from keras_datareaders.ClassificationReader import ClassificationReader
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
import math
from models.keras.cifar import densenet
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.utils import to_categorical

GPU_NUM = 2
EPOCH_NUM = 100
densenet_depth = 40
densenet_growth_rate = 12

def lr_schedule(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate

session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
K.set_session(session)
model = densenet(depth=densenet_depth,
                 growth_rate = densenet_growth_rate)
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

train_generator = train_dataGen.flow_from_directory(directory="../data/cifar10/train", target_size=(32,32), batch_size=100,
                                                    class_mode='categorical')
probar = ProgressBarCallback()
es = EarlyStopping(monitor='val_acc', patience=EPOCH_NUM)
checkpoint = ModelCheckpoint(filepath="cifar10_densenet.h5", save_best_only=True, save_weights_only=True)
lrate = LearningRateScheduler(lr_schedule)

reader = ClassificationReader()
x_test, y_test = reader.readData(phrase="test")
y_test = to_categorical(y_test, num_classes=10)

model.fit_generator(generator=train_generator, steps_per_epoch=50000/100,
                    epochs=EPOCH_NUM, verbose=0,
                    validation_data=(x_test, y_test),validation_steps=10000/100,
                    callbacks=[probar,es,checkpoint, lrate])

# scores = model.evaluate(x_test, y_test, verbose=1, steps=EPOCH_NUM)
# print('Test loss:', scores[0])
# print('Test accuracy:', scores[1])
