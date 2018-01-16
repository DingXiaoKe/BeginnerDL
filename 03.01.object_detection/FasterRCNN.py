from keras_datareaders.pascal_voc_parser import get_data, get_anchor_gt
from keras.layers import ZeroPadding2D, Convolution2D, Activation, MaxPooling2D, \
    Input, TimeDistributed, Flatten,Dense
from keras_layers.frrcnn import FixedBatchNormalization, conv_block, identity_block, \
    RoiPoolingConv, classifier_layers
from keras.optimizers import Adam
from keras.models import Model
from keras_losses.frrcnn import rpn_loss_cls, rpn_loss_regr, class_loss_cls, class_loss_regr
import keras.backend as K
from keras_commons.Config import Config
from keras.utils import generic_utils
from keras_commons.roi_helpers import rpn_to_roi, calc_iou
import numpy as np
import random
import time
def baseNetwork(imageInput):
    bn_axis = 3
    network = ZeroPadding2D((3,3))(imageInput)
    network = Convolution2D(filters=64, kernel_size=(7,7), strides=(2,2), trainable=True)(network)
    network = FixedBatchNormalization(axis=bn_axis)(network)
    network = Activation('relu')(network)
    network = MaxPooling2D((3,3), strides=(2,2))(network)

    network = conv_block(network, kernel_size=3, filters=[64,64,256], stage=2, block='a', strides=(1,1), trainable=True)
    network = identity_block(network, 3, [64,64,256], stage=2, block='b', trainable=True)
    network = identity_block(network, 3, [64,64,256], stage=2, block='c', trainable=True)

    network = conv_block(network, 3, [128, 128, 512], stage=3, block='a', trainable=True)
    network = identity_block(network, 3, [128, 128, 512], stage=3, block='b', trainable=True)
    network = identity_block(network, 3, [128, 128, 512], stage=3, block='c', trainable=True)
    network = identity_block(network, 3, [128, 128, 512], stage=3, block='d', trainable=True)

    network = conv_block(network, 3, [256, 256, 1024], stage=4, block='a', trainable=True)
    network = identity_block(network, 3, [256, 256, 1024], stage=4, block='b', trainable=True)
    network = identity_block(network, 3, [256, 256, 1024], stage=4, block='c', trainable=True)
    network = identity_block(network, 3, [256, 256, 1024], stage=4, block='d', trainable=True)
    network = identity_block(network, 3, [256, 256, 1024], stage=4, block='e', trainable=True)
    network = identity_block(network, 3, [256, 256, 1024], stage=4, block='f', trainable=True)

    return network


def rpn_layer(base_layers, num_anchors):
    network = Convolution2D(512, (3,3), padding="same", activation='relu', kernel_initializer='normal')(base_layers)
    r_class = Convolution2D(num_anchors, (1,1), activation='sigmoid', kernel_initializer='uniform')(network)
    r_regr = Convolution2D(num_anchors * 4, (1,1), activation='linear', kernel_initializer='zero')(network)

    return [r_class, r_regr, base_layers]


def classifier_layer(base_layer, input_rois, num_rois, nb_classes=21):
    pooling_regions = 14
    input_shape = (num_rois,14,14,1024)

    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layer, input_rois])
    out = classifier_layers(out_roi_pool, input_shape=input_shape, trainable=True)
    out = TimeDistributed(Flatten())(out)

    out_class = TimeDistributed(Dense(nb_classes, activation='softmax', kernel_initializer='zero'))(out)
    out_regr = TimeDistributed(Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'))(out)

    return [out_class, out_regr]

config = Config()

all_imgs, classes_count, class_mapping = get_data("../data/")


image_input = Input(shape=(None, None, 3))
shared_layers = baseNetwork(image_input)
num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)
rpn = rpn_layer(shared_layers, num_anchors)

roi_input = Input(shape=(config.num_rois, 4))
classifer = classifier_layer(shared_layers, roi_input, config.num_rois, nb_classes=len(classes_count)) # 得到的数据没有背景的分类

model_rpn = Model(image_input, rpn[:2])
model_classifier = Model([image_input, roi_input], classifer)
model_all = Model([image_input, roi_input], rpn[:2] + classifer)



train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

data_gen_train = get_anchor_gt(train_imgs, classes_count, config, mode='train')
data_gen_val = get_anchor_gt(val_imgs, classes_count, config,  mode='val')


optimizer = Adam(lr=1e-4)
optimizer_classifier = Adam(lr=1e-4)
model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier, loss=[class_loss_cls, class_loss_regr(len(classes_count)-1)],
                         metrics=['acc'])
model_all.compile(optimizer='sgd', loss='mae')

iter_num = 0
epoch_length = 2000
losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

for epoch_num in range(2000):

    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, 2000))

    while True:
        try:
            if len(rpn_accuracy_rpn_monitor) == epoch_length and config.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            X, Y, img_data = next(data_gen_train)

            loss_rpn = model_rpn.train_on_batch(X, Y)

            P_rpn = model_rpn.predict_on_batch(X)

            R = rpn_to_roi(P_rpn[0], P_rpn[1], config, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)

            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            X2, Y1, Y2 = calc_iou(R, img_data, config, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if config.num_rois > 1:
                if len(pos_samples) < config.num_rois/2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, int(config.num_rois/2), replace=False).tolist()
                try:
                    selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples), replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, config.num_rois - len(selected_pos_samples), replace=True).tolist()

                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            progbar.update(iter_num, [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                      ('detector_cls', np.mean(losses[:iter_num, 2])), ('detector_regr', np.mean(losses[:iter_num, 0]))])

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if config.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if config.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights(config.model_path)

                break

        except Exception as e:
            print('Exception: {}'.format(e))
            continue

print('Training complete, exiting.')








