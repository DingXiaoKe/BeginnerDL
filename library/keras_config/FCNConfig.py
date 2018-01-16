from keras_config.Config import Config
import os

class FCNConfig(Config):
    def __init__(self):
        super(FCNConfig, self).__init__()
        self.GPU_NUM = 1
        self.WEIGHT_DECAY = 0.0001/2
        self.BATCH_MOMENTUM = 0.95
        self.BATCH_SIZE = 16
        self.IMAGE_SIZE = 320
        self.NUM_CLASS = 21
        self.EPOCH_NUM = 32
        self.VOC_CLASSES = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                            'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                            'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
                            'Sheep', 'Sofa', 'Train', 'Tvmonitor']
        self.TRAIN_FILE_PATH = os.path.join(self.DATAPATH, "VOC2012", "ImageSets", "Segmentation", "train.txt")
        self.VAL_FILE_PATH = os.path.join(self.DATAPATH, "VOC2012", "ImageSets", "Segmentation", "val.txt")
        self.DATA_DIR = os.path.join(self.DATAPATH, "VOC2012", "JPEGImages")
        self.LABEL_DIR = os.path.join(self.DATAPATH, "VOC2012", "SegmentationClass")
        self.LR_BASE = 0.01 * (float(self.BATCH_SIZE) / 16)
        self.LR_POWER = 0.9
        self.LABEL_CVAL = 255