import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import copy
from six.moves import xrange

class pascal_voc(object):
    def __init__(self, batch_size, image_data_generator, VOCPath='../data', rebuild=False):
        self.data_path = os.path.join(VOCPath, 'VOC2007')
        self.image_size = 448
        self.cell_size = 7
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
                        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                        'train', 'tvmonitor']
        self.class_to_ind = dict(zip(self.classes, xrange(len(self.classes))))
        self.flipped = True
        self.rebuild = rebuild
        self.cursor = 0
        self.epoch = 1
        self.gt_labels = None
        self.batch_size = batch_size
        self.image_data_generator = image_data_generator

    def __next__(self):
        gt_labels = self.gt_labels
        images = np.zeros((self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros((self.batch_size, self.cell_size, self.cell_size, 25))
        count = 0
        while count < self.batch_size:
            imname = gt_labels[self.cursor]['imname']
            flipped = gt_labels[self.cursor]['flipped']
            image = self.image_read(imname, flipped)
            image = self.image_data_generator.random_transform(image, seed=1000)
            images[count, :, :, :] =image
            labels[count, :, :, :] = gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(gt_labels):
                np.random.shuffle(gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, np.reshape(labels, newshape=(self.batch_size, self.cell_size * self.cell_size * 25))

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        if flipped:
            image = image[:, ::-1, :]
        return image

    def prepare(self, model):
        gt_labels = self.load_labels(model)
        if self.flipped:
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] = gt_labels_cp[idx]['label'][:, ::-1, :]
                # print(gt_labels_cp[idx]['label'], '   ', gt_labels_cp[idx]['label'][:, ::-1, :])
                for i in xrange(self.cell_size):
                    for j in xrange(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            gt_labels_cp[idx]['label'][i, j, 1] = self.image_size - 1 - gt_labels_cp[idx]['label'][i, j, 1]
            gt_labels += gt_labels_cp
        np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels

    def load_labels(self, model):
        if model == 'train':
            txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        if model == 'test':

            txtname = os.path.join(self.data_path, 'ImageSets', 'Main', 'test.txt')

        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]

        gt_labels = []
        for index in self.image_index:
            label, num = self.load_pascal_annotation(index)
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append({'imname': imname, 'label': label, 'flipped': False})
        return gt_labels

    def load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC format."""

        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(imname)
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])

        label = np.zeros((self.cell_size, self.cell_size, 25))
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            x1 = max(min((float(bbox.find('xmin').text)) * w_ratio, self.image_size), 0)
            y1 = max(min((float(bbox.find('ymin').text)) * h_ratio, self.image_size), 0)
            x2 = max(min((float(bbox.find('xmax').text)) * w_ratio, self.image_size), 0)
            y2 = max(min((float(bbox.find('ymax').text)) * h_ratio, self.image_size), 0)
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            '''
            boxes的四个坐标的含义是
            1. Group Truce边框的中心点的x坐标
            2. Group Truce边框的中心点的y坐标
            3. Group Truce边框的宽度
            4. Group Truce边框的高度
            '''
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            '''
            由于YoLo V1算法规定，如果一个格子正好包含GroupTruce的中心点，那么这个格子就负责预测这个GroupTruce对应的物体
            由于boxes的前两列是Group Truce边框的中心点，所以(x_ind,y_ind)就是在7*7的图片上，当前GroupTruce中心点对应的
            位置，所以(x_ind,y_ind)这个位置也必然负责当前GroupTruce的预测，所以就有label[y_ind, x_ind, 0] = 1
            '''
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(objs)
