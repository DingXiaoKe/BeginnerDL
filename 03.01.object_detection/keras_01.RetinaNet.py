import os

import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from keras.utils.vis_utils import plot_model

from lib.utils.progressbar.keras.ProgressBarCallback import ProgressBar
from lib.models.layers.keras.retinanet import ResNet50RetinaNet
from lib.models.losses.keras.retinanet import smooth_l1,focal
from lib.datareader.keras.pascal_voc import PascalVocGenerator


def newModel(image_input):
    return ResNet50RetinaNet(image_input, num_classes=20)

GPU_NUM = 1
EPOCH_NUM = 100

session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
keras.backend.set_session(session)
image_input = keras.layers.Input(shape=(None,None,3))
model = newModel(image_input)
if GPU_NUM >= 2:
    model = multi_gpu_model(model, gpus=GPU_NUM)

plot_model(model,to_file='RetinaNet.png',show_shapes=True,show_layer_names=True)
model.compile(
    loss={
        'regression'    : smooth_l1(),
        'classification': focal()
    },
    optimizer=keras.optimizers.adam(lr=1e-5, clipnorm=0.001)
)

# create image data generator objects
train_image_data_generator = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
)
val_image_data_generator = keras.preprocessing.image.ImageDataGenerator()
batch_size = 1
# create a generator for training data
train_generator = PascalVocGenerator(
    "../data/VOC2007",
    'trainval',
    train_image_data_generator,
    batch_size=batch_size
)

# create a generator for testing data
val_generator = PascalVocGenerator(
    "../data/VOC2007",
    'test',
    val_image_data_generator,
    batch_size=batch_size
)

probar = ProgressBar()

# start training

model.fit_generator(
    generator=train_generator,
    steps_per_epoch=len(train_generator.image_names) // batch_size,
    epochs=EPOCH_NUM,
    verbose=0,
    validation_data=val_generator,
    validation_steps=3000,
    callbacks=[
        keras.callbacks.ModelCheckpoint(os.path.join('snapshots', 'retina_resnet50_coco_best.h5'), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
        probar
    ],
)

# store final result too
model.save_weights("retinaNet.h5")
