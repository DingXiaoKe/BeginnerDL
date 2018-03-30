# coding=utf-8
import numpy as np
import pandas as pd
import os

from torch.utils.data import Dataset
from PIL import Image

from lib.datareader.common import read_image_bgr

class FacesGenderDataSet(Dataset):
    def __init__(self, data_path='../ganData/face/celebA/', attr_file_path = "../common/face_attr.csv",
                 transform=None, target_transform=None, image_size=96):
        super(FacesGenderDataSet, self).__init__()
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.data = pd.read_csv(attr_file_path, usecols=('File_Names', 'Male')).as_matrix()

    def __getitem__(self, index):
        record = self.data[index]
        filepath = os.path.join(self.data_path, record[0])
        label = int(record[1])

        label = 0 if label == -1 else 1

        img = read_image_bgr(filepath, self.image_size, self.image_size).astype(np.uint8)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self):
        return 20000#self.data.shape[0]



