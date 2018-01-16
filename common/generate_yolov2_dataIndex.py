import os
import xml.etree.ElementTree as ET
PATH = "../data/VOC2007"
MAIN_PATH = os.path.join(PATH, "ImageSets", "Main")
TRAIN_FILE_PATH = os.path.join(MAIN_PATH, "trainval.txt")
TEST_FILE_PATH = os.path.join(MAIN_PATH, "test.txt")

JPEG_FILE_PATH = os.path.join(PATH, "JPEGImages")
ANNO_FILE_PATH = os.path.join(PATH, "Annotations")

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
           "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

if __name__ == '__main__':
    fileList = [[TRAIN_FILE_PATH, "train"], [TEST_FILE_PATH, "test"]]
    for rec in fileList:
        with open(rec[0], 'r') as file:
            indexList = file.read().strip().split()

        with open("%s.txt" % rec[1], 'w') as file:
            for index, ImageIndex in enumerate(indexList):
                image_file_path = os.path.join(JPEG_FILE_PATH, "%s.jpg" % ImageIndex)
                anno_file_path = os.path.join(ANNO_FILE_PATH, "%s.xml" % ImageIndex)
                tree=ET.parse(anno_file_path)
                root = tree.getroot()
                size = root.find('size')
                w = int(size.find('width').text)
                h = int(size.find('height').text)

                for obj in root.iter('object'):
                    difficult = obj.find('difficult').text
                    cls = obj.find('name').text
                    if cls not in classes or int(difficult) == 1:
                        continue
                    cls_id = classes.index(cls)
                    xmlbox = obj.find('bndbox')
                    xmin = xmlbox.find('xmin').text
                    ymin = xmlbox.find('ymin').text
                    xmax = xmlbox.find('xmax').text
                    ymax = xmlbox.find('ymax').text
                    width = float(xmax) - float(xmin)
                    height = float(ymax) - float(ymin)
                    #images/person.jpg, 178, 52, 110, 370, 0
                    file.write("%s, %s, %s, %d, %d, %d\r" % (image_file_path, xmin, ymin, int(width), int(height), cls_id))