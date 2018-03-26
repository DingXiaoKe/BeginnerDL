# coding=utf-8
from lib.config.Config import Config

class STLConfig(Config):
    def __init__(self):
        super(STLConfig, self).__init__()
        self.NUM_OUTPUTS = 10
        self.GPU_NUM = 1
        self.BATCH_SIZE = 100
        self.IMAGE_SIZE = 96
        self.DATA_PATH = "../data/STL/"
        self.GAN_DATA_PATH = "../ganData/STL/"
        self.NUM_WORKERS_LOAD_IMAGE = 4
        self.IMAGE_CHANNEL = 3
        self.BATCH_SIZE = 50
        self.EPOCH_NUM = 1000
        self.LR_GENERATOR = 2e-4
        self.LR_DISCRIMINATOR = 2e-4
        self.BETA1 = 0.5
        self.GPU_NUM = 2
        self.NOISE_Z = 100

        self.GENERATOR_FEATURES_NUM = 64
        self.DISCRIMINATOR_FEATURES_NUM = 64
        self.D_EVERY = 1  # 每1个batch训练一次判别器
        self.G_EVERY = 5  # 每5个batch训练一次生成器
        self.DECAY_EVERY = 10  # 没10个epoch保存一次模型
        self.SAVE_PATH = "output/"

        self.LABELS = ["airplane",
                       "bird",
                       "car",
                       "cat",
                       "deer",
                       "dog",
                       "horse",
                       "monkey",
                       "ship",
                       "truck"]