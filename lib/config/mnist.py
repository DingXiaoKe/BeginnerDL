from config.Config import Config

class MNISTConfig(Config):
    def __init__(self):
        super(MNISTConfig, self).__init__()
        self.NUM_OUTPUTS = 10
        self.GPU_NUM = 1
        self.BATCH_SIZE = 32
        self.IMAGE_SIZE = 28
        self.IMAGE_CHANNEL = 1
        self.DATAPATH = "../data/mnist.npz"
        self.ENCODED_DIM = 100
        self.EPOCH_NUM = 30000
        self.SAVE_INTERVAL = 200
        self.LEARNINGRATE_LR = 0.0002
        self.LEARNINGRATE_BETA_1 = 0.5
