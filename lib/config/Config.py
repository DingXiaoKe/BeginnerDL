class Config(object):
    def __init__(self):
        self.LEAKEY_ALPHA = 0.1
        self.KEEP_PROB = 0.5
        self.GPU_NUM = 2
        self.EPOCH_NUM = 20
        self.BATCH_SIZE = 100
        self.NUM_OUTPUTS = 10
        self.DATAPATH = "../data"

        self.IMAGE_CHANNEL = 3