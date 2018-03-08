from torch.utils.data import Dataset
import os
from keras_datareaders import mnistReader as reader
from PIL import Image
import torch

class MNISTDataSet(Dataset):
    def __init__(self, root="../data/mnist.npz", train=True, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        (self.train_data, self.train_labels),(self.test_data, self.test_labels) = reader.read_mnist(root)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='L')

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