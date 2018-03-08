# coding=utf-8
import numpy as np
from PIL import Image

def read_mnist(path='../data/mnist.npz'):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)

def read_image_bgr(path):
    image = np.asarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()