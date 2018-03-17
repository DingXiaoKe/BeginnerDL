import tensorflow as tf
import keras.backend as K
import numpy as np

from keras.utils import multi_gpu_model
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

from lib.config.FCNConfig import FCNConfig
from lib.models.losses.keras.fcn import softmax_sparse_crossentropy_ignoring_last_label, sparse_accuracy_ignoring_last_label
from lib.datareader.keras.fcn_generator import SegDataGenerator
from lib.models.layers.keras.FCN import FCN_Resnet50_16s
from lib.utils.progressbar.ProgressBar import ProgressBar

cfg = FCNConfig()

def lr_scheduler(epoch, mode='power_decay'):
    '''if lr_dict.has_key(epoch):
        lr = lr_dict[epoch]
        print 'lr: %f' % lr'''

    if mode is 'power_decay':
        # original lr scheduler
        lr = cfg.LR_BASE * ((1 - float(epoch)/cfg.EPOCH_NUM) ** cfg.LR_POWER)
    if mode is 'exp_decay':
        # exponential decay
        lr = (float(cfg.LR_BASE) ** float(cfg.LR_POWER)) ** float(epoch+1)
    # adam default lr
    if mode is 'adam':
        lr = 0.001

    if mode is 'progressive_drops':
        # drops as progression proceeds, good for sgd
        if epoch > 0.9 * cfg.EPOCH_NUM:
            lr = 0.0001
        elif epoch > 0.75 * cfg.EPOCH_NUM:
            lr = 0.001
        elif epoch > 0.5 * cfg.EPOCH_NUM:
            lr = 0.01
        else:
            lr = 0.1

    print('lr: %f' % lr)
    return lr

def get_file_len(file_path):
    fp = open(file_path)
    lines = fp.readlines()
    fp.close()
    return len(lines)

session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
K.set_session(session)
model = FCN_Resnet50_16s()
if cfg.GPU_NUM >= 2:
    model = multi_gpu_model(model, gpus=cfg.GPU_NUM)

model.compile(loss=softmax_sparse_crossentropy_ignoring_last_label,
              optimizer=SGD(lr=cfg.LR_BASE, momentum=0.9),
              metrics=[sparse_accuracy_ignoring_last_label])

scheduler = LearningRateScheduler(lr_scheduler)

train_datagen = SegDataGenerator(zoom_range=[0.5, 2.0],
                                 zoom_maintain_shape=True,
                                 crop_mode='random',crop_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                                 rotation_range=0, shear_range=0,
                                 horizontal_flip=True,channel_shift_range=20,
                                 fill_mode='constant', label_cval=cfg.LABEL_CVAL)
val_datagen = SegDataGenerator()
steps_per_epoch = int(np.ceil(get_file_len(cfg.TRAIN_FILE_PATH) / float(cfg.BATCH_SIZE)))

probar = ProgressBar()
model.fit_generator(generator=train_datagen.flow_from_directory(
                        file_path=cfg.TRAIN_FILE_PATH,
                        data_dir=cfg.DATA_DIR,
                        data_suffix= ".jpg",label_dir=cfg.LABEL_DIR,
                        label_suffix= ".png",classes= cfg.NUM_CLASS,
                        target_size=(cfg.IMAGE_SIZE,cfg.IMAGE_SIZE), color_mode='rgb',
                        batch_size=cfg.BATCH_SIZE,shuffle=True,
                        loss_shape=None,
                        ignore_label=255
                    ),
                    steps_per_epoch=steps_per_epoch,
                    epochs=cfg.EPOCH_NUM,
                    verbose=0,workers=4,class_weight=None,
                    callbacks=[scheduler, probar])
model.save_weights("FCN.h5")
print("Saved model to disk")