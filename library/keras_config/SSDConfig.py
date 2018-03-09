from config.Config import Config

class SSDConfig(Config):
    def __init__(self):
        super(SSDConfig, self).__init__()
        self.GPU_NUM = 1
        self.BATCH_SIZE = 16
        self.IMAGE_SIZE = 300
        self.NUM_CLASS = 21
        self.EPOCH_NUM = 32
        self.PRIORBOX = [
            {'layer_width': 38, 'layer_height': 38, 'num_prior': 3, 'min_size':  30.0, 'max_size': None,  'aspect_ratios': [1.0, 2.0, 1/2.0]},
            {'layer_width': 19, 'layer_height': 19, 'num_prior': 6, 'min_size':  60.0, 'max_size': 114.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
            {'layer_width': 10, 'layer_height': 10, 'num_prior': 6, 'min_size': 114.0, 'max_size': 168.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
            {'layer_width':  5, 'layer_height':  5, 'num_prior': 6, 'min_size': 168.0, 'max_size': 222.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
            {'layer_width':  3, 'layer_height':  3, 'num_prior': 6, 'min_size': 222.0, 'max_size': 276.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
            {'layer_width':  1, 'layer_height':  1, 'num_prior': 6, 'min_size': 276.0, 'max_size': 330.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
        ]
        self.VOC_CLASSES = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
                                          'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
                                          'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
                                          'Sheep', 'Sofa', 'Train', 'Tvmonitor']