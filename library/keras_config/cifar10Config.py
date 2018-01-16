from keras_config.Config import Config

class Cifar10Config(Config):
    def __init__(self):
        super(Cifar10Config, self).__init__()
        self.NUM_OUTPUTS = 10
        self.GPU_NUM = 1
        self.BATCH_SIZE = 100
        self.IMAGE_SIZE = 32
        self.LABELS = ["airplane",
                       "automobile",
                       "bird",
                       "cat",
                       "deer",
                       "dog",
                       "frog",
                       "horse",
                       "ship",
                       "truck"]
