# coding=utf-8
from torch.utils.data import Dataset
from lib.datareader.DataReaderForClassification import DataReader
from PIL import Image
import numpy as np

class Cifar10DataSet(Dataset):
    def __init__(self, root="/data/cifar10/", subFolder="", train=True,transform=None, target_transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        reader = DataReader(self.root)
        if train == True:
            self.train_data, self.train_label = reader.readData("train", subFolder=subFolder, image_shape=(32,32,3))
        else:
            self.test_data, self.test_label = reader.readData("test", subFolder=subFolder, image_shape=(32,32,3))

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_label[index]
        else:
            img, target = self.test_data[index], self.test_label[index]

        img = Image.fromarray(img.astype(np.uint8))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)