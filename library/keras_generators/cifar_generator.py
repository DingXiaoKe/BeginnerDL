import os
import random
import threading

import keras
import numpy as np

from keras_commons.image import read_image_bgr

cifar10_classes = {
    'airplane' : 0,
    'automobile' : 1,
    'bird' : 2,
    'cat' : 3,
    'deer' : 4,
    'dog' : 5,
    'frog' : 6,
    'horse' : 7,
    'ship' : 8,
    'truck' : 9
}

class ClassificationGenerator(object):
    def __init__(self, imageDataGenerator, phrase, datapath, image_shape=(32, 32,3), batch_size=100, classes=cifar10_classes, imageExt = ".jpg", **kwargs):
        self.imageDataGenerator = imageDataGenerator
        self.phrase = phrase
        self.datapath = datapath
        self.image_names = [l.strip() for l in open(os.path.join(self.datapath, '%s.txt' % self.phrase)).readlines()]
        self.image_folder = os.path.join(self.datapath, phrase)
        self.classes = classes
        self.batchSize = batch_size
        self.imageShape = image_shape

        self.lock = threading.Lock()
    def __next__(self):
        return self.next()

    def next(self):
        with self.lock:
            list = self.randomSelect()
            imageBatch = np.zeros(shape=(self.batchSize, ) + self.imageShape, dtype=keras.backend.floatx())
            labelBatch = np.zeros(shape=(self.batchSize, len(self.classes)), dtype=np.int32)
            for index, image_name_index in enumerate(list):
                path = os.path.join(self.image_folder, self.image_names[image_name_index])
                image = read_image_bgr(path)
                image = image.reshape(self.imageShape)
                image = self.imageDataGenerator.random_transform(image, seed=10)
                imageBatch[index] = image
                split = self.image_names[image_name_index].split('.')[0].split('_')
                label = keras.utils.to_categorical(split[len(split) - 1], len(self.classes))
                labelBatch[index] = label
        return imageBatch, labelBatch

    def randomSelect(self):
        list = range(len(self.image_names))
        return random.sample(list, self.batchSize)