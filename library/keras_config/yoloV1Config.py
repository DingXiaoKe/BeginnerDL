from keras_config.Config import Config

class YoloV1Config(Config):
    def __init__(self):
        super(YoloV1Config, self).__init__()
        self.NUM_OUTPUTS = 1470
        self.GPU_NUM = 1
        self.BATCH_SIZE = 3
        self.IMAGE_SIZE = 448
        self.CELL_SIZE = 7
        self.BOXES_PER_CELL = 2
        self.OBJECT_SCALE = 1.0
        self.NOOBJECT_SCALE = 1.0
        self.CLASS_SCALE = 2.0
        self.COORD_SCALE = 5.0
        self.NUM_CLASS = 20
        self.BATCH_SIZE = 3
        self.IMAGE_SIZE = 448
        self.BOUNDARY1 = self.CELL_SIZE * self.CELL_SIZE * self.NUM_CLASS
        self.BOUNDARY2 = self.BOUNDARY1 + self.CELL_SIZE * self.CELL_SIZE * self.BOXES_PER_CELL