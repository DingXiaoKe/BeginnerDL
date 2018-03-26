# coding=utf-8
from lib.config.Config import Config

class AnimeHeadConfig(Config):
    def __init__(self):
        super(AnimeHeadConfig, self).__init__()
        self.DATA_PATH = "../data/faces/"
        self.NUM_WORKERS_LOAD_IMAGE = 4
        self.IMAGE_CHANNEL = 3
        self.IMAGE_SIZE = 96
        self.BATCH_SIZE = 256
        self.EPOCH_NUM = 200
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
