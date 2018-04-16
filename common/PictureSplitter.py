# coding=utf-8
from PIL import Image
import numpy as np
import glob
from random import randint, shuffle
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
DATA_PATH = os.path.join("..", "ganData")
DATA_NAMES = "edges2handbags"
DIRECTION = 1

loadSize = 268
imageSize = 256

def read_image(fn, direction=0):
    im = Image.open(fn)
    im = im.resize( (loadSize*2, loadSize), Image.BILINEAR )
    arr = np.array(im)/255*2-1
    w1,w2 = (loadSize-imageSize)//2,(loadSize+imageSize)//2
    h1,h2 = w1,w2
    imgA = arr[h1:h2, loadSize+w1:loadSize+w2, :]
    imgB = arr[h1:h2, w1:w2, :]
    if randint(0,1):
        imgA=imgA[:,::-1]
        imgB=imgB[:,::-1]
    if False:
        imgA = np.moveaxis(imgA, 2, 0)
        imgB = np.moveaxis(imgB, 2, 0)
    if direction==0:
        return imgA, imgB
    else:
        return imgB,imgA

PHRASE = ["train", "val", "test"]

DATA_NAMES_NEW = "%s_%s" % (DATA_NAMES, "fixed")

DATA_NAMES_PATH = os.path.join(DATA_PATH, DATA_NAMES)
DATA_NAMES_NEW_PATH = os.path.join(DATA_PATH, DATA_NAMES_NEW)

for phrase in PHRASE:
    DATA_NAMES_PHRASE_PATH = os.path.join(DATA_NAMES_PATH, phrase)
    DATA_NAMES_PHRASE_NEW_PATH = os.path.join(DATA_NAMES_NEW_PATH, phrase)

    if not os.path.exists(DATA_NAMES_NEW_PATH):
        os.mkdir(DATA_NAMES_NEW_PATH)

    if not os.path.exists(DATA_NAMES_PHRASE_NEW_PATH):
        os.mkdir(DATA_NAMES_PHRASE_NEW_PATH)

    PATH_A = os.path.join(DATA_NAMES_PHRASE_NEW_PATH, "A")
    PATH_B = os.path.join(DATA_NAMES_PHRASE_NEW_PATH, "B")

    if not os.path.exists(PATH_A):
            os.mkdir(PATH_A)

    if not os.path.exists(PATH_B):
        os.mkdir(PATH_B)

    fileList = glob.glob(os.path.join(DATA_NAMES_PHRASE_PATH, "*.jpg"))
    pbar = tqdm(enumerate(fileList))
    for index, file in pbar:
        pbar.set_description("loading images : %s" % (len(fileList) - index - 1))
        imgA,imgB = read_image(file, direction=DIRECTION)
        int_X = ( (imgA+1)/2*255).clip(0,255).astype('uint8')
        int_X = int_X.reshape(imageSize,imageSize, 3)
        plt.imsave(os.path.join(PATH_A, "%s.jpg" % index), int_X)

        int_X = ( (imgB+1)/2*255).clip(0,255).astype('uint8')
        int_X = int_X.reshape(imageSize,imageSize, 3)
        plt.imsave(os.path.join(PATH_B, "%s.jpg" % index), int_X)
