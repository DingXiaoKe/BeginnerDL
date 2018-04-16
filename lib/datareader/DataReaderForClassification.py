# coding=utf-8
'''
此类可以适用于Cifar10，Cifar20，Cifar100，flowers，STL数据集的读取
'''
import os
import numpy as np
from lib.datareader.common import read_image_bgr
from tqdm import tqdm
class DataReader(object):
    def __init__(self, dataPath="/data/cifar10"):
        self.dataPath = dataPath

    def readData(self, phrase="train", subFolder="", image_shape=(32,32,3)):
        imagelist = []
        labellist = []
        total_count = 0

        for rt, dirs, files in os.walk(os.path.join(self.dataPath, phrase, "" if subFolder == "" else subFolder)):
            if dirs == []:
                list = self._getImage(os.path.join(self.dataPath, phrase, subFolder))
                total_count += len(list)
                imagelist.append(list)
                labellist.append([subFolder] * len(list))
            else:
                for directory in dirs:
                    list = self._getImage(os.path.join(self.dataPath, phrase, directory))
                    total_count += len(list)
                    imagelist.append(list)
                    labellist.append([directory] * len(list))
                break


        imagelist = np.reshape(imagelist, newshape=(1, total_count))[0]
        labellist = np.reshape(labellist, newshape=(1, total_count))[0]

        reImageList = np.zeros(shape=(total_count,) + image_shape, dtype=np.float32)
        reLabelList = np.zeros(shape=(total_count, 1), dtype=np.int32)
        text = ""
        pbar = tqdm(enumerate(imagelist))
        for index, image in pbar:
            pbar.set_description("loading images : %s" % (total_count - index - 1))
            label = labellist[index]
            imageData = read_image_bgr(os.path.join(self.dataPath, phrase, label, image), image_shape[0], image_shape[1])
            imageData = imageData.astype(np.uint8)
            reImageList[index] = imageData
            reLabelList[index] = label

        return reImageList, reLabelList
    def _getImage(self, folder):
        imageNameList = []
        for rt, dirs, files in os.walk(folder):
            for f in files:
                if f.endswith(".jpg"):
                    imageNameList.append(f)
            break

        return imageNameList