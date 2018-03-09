from config.Config import Config
import numpy as np

class YoloV2Config(Config):
    def __init__(self):
        super(YoloV2Config, self).__init__()

        self.GPU_NUM = 1
        self.BATCH_SIZE = 3
        self.IMAGE_SIZE = 416
        self.L2 = 0.0005
        self.BOX_CELLS_NUM = 32
        self.BOUX_CELL_SIZE = self.IMAGE_SIZE//self.BOX_CELLS_NUM

        self.ANCHOR_VALUE = np.array(
            [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]],dtype='float32')
        self.ANCHOR_LENGTH = len(self.ANCHOR_VALUE)
        self.TRAIN_FILE = "../data/VOC2007/train.txt"
        self.TEST_FILE = "../data/VOC2007/test.txt"

        self.NUM_OUTPUTS = 125
        self.CLASS_NAME = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
                                     "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
                                     "sheep", "sofa", "train", "tvmonitor"]
        self.NUM_CLASSES = len(self.CLASS_NAME)
