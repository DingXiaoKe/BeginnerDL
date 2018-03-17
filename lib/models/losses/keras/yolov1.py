import numpy as np
import tensorflow as tf
from config.yoloV1Config import YoloV1Config

def loss_layer(y_true, y_pred):

    #y_pred : (?, 1470)

    '''
    :param y_true: (?, 1470)
    1470 = 7 * 7 * (20 + (5 * 2))
    :param y_pred: (?, 1225)
    1225 = 7 * 7 * (20 + 5)
    一张图片被分为7 * 7的格子，每个格子里面的数据是25维的，其中包括1维的confidence，4维的坐标信息，以及20维的分类one-hot
    编码(VOC有20类)
    :return:
    '''
    cfg = YoloV1Config()

    predicts = y_pred #由卷机网络输出的结果，(?,1470)
    labels = tf.reshape(y_true,[cfg.BATCH_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE, 25]) #真实值，(?,7,7,25)

    predict_classes = tf.reshape(predicts[:, :cfg.BOUNDARY1], [cfg.BATCH_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.NUM_CLASS]) # 预测的类别 ？，7，7，20
    predict_scales = tf.reshape(predicts[:, cfg.BOUNDARY1:cfg.BOUNDARY2], [cfg.BATCH_SIZE,cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL] ) # 预测的信心 7，7，2
    predict_boxes = tf.reshape(predicts[:, cfg.BOUNDARY2:], [cfg.BATCH_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL, 4])# 预测的boundingBox (?, 7，7，2, 4)

    response = tf.reshape(labels[:, :, :, 0], [cfg.BATCH_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE, 1]) # 0 或者 1 # 当前格子是否负责预测物体
    boxes = tf.reshape(labels[:, :, :, 1:5], [cfg.BATCH_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE, 1, 4]) # group truce的Bounding Box信息，注意坐标是中心点
    # 这里为何要在第3维上进行扩充呢？因为predict包含的是两个边界盒子，而label里面包含的是1个边界盒子，所以需要扩充一个边界盒子
    boxes = tf.tile(boxes, [1, 1, 1, cfg.BOXES_PER_CELL, 1]) / cfg.IMAGE_SIZE
    classes = labels[:, :, :, 5:] # 真实值的类别

    offset = np.transpose(np.reshape(np.array([np.arange(cfg.CELL_SIZE)] * cfg.CELL_SIZE * cfg.BOXES_PER_CELL),
                                 (cfg.BOXES_PER_CELL, cfg.CELL_SIZE, cfg.CELL_SIZE)), (1, 2, 0))

    offset = tf.constant(offset, dtype=tf.float32)
    offset = tf.reshape(offset, [1, cfg.CELL_SIZE, cfg.CELL_SIZE, cfg.BOXES_PER_CELL])
    offset = tf.tile(offset, [cfg.BATCH_SIZE, 1, 1, 1]) # (?,7,7,2)

    predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / cfg.CELL_SIZE,
                                   (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / cfg.CELL_SIZE,
                                   tf.square(predict_boxes[:, :, :, :, 2]), #取平方
                                   tf.square(predict_boxes[:, :, :, :, 3])]) # 取平方 # (4,?,7,7,2)

    predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0]) # (?, 7,7,2,4)

    iou_predict_truth = calc_iou(predict_boxes_tran, boxes) # boxes : ?,7,7,2,4

    # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
    object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

    # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
    noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

    boxes_tran = tf.stack([boxes[:, :, :, :, 0] * cfg.CELL_SIZE - offset,
                           boxes[:, :, :, :, 1] * cfg.CELL_SIZE - tf.transpose(offset, (0, 2, 1, 3)),
                           tf.sqrt(boxes[:, :, :, :, 2]),
                           tf.sqrt(boxes[:, :, :, :, 3])])
    boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

    # class_loss , MSELoss
    class_delta = response * (predict_classes - classes) # (?,7,7,1) * (?,7,7,20)
    class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * cfg.CLASS_SCALE # 只在1，2，3维上做，不可以在batch上做呀

    # object_loss, MSELoss
    object_delta = object_mask * (predict_scales - iou_predict_truth)
    object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss') * cfg.OBJECT_SCALE

    # noobject_loss
    noobject_delta = noobject_mask * predict_scales
    noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * cfg.NOOBJECT_SCALE

    # coord_loss
    coord_mask = tf.expand_dims(object_mask, 4)
    boxes_delta = coord_mask * (predict_boxes - boxes_tran)
    coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss') * cfg.COORD_SCALE

    tf.losses.add_loss(class_loss)
    tf.losses.add_loss(object_loss)
    tf.losses.add_loss(noobject_loss)
    tf.losses.add_loss(coord_loss)

    tf.summary.scalar('class_loss', class_loss)
    tf.summary.scalar('object_loss', object_loss)
    tf.summary.scalar('noobject_loss', noobject_loss)
    tf.summary.scalar('coord_loss', coord_loss)

    tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
    tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
    tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
    tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
    tf.summary.histogram('iou', iou_predict_truth)
    return tf.losses.get_total_loss()

'''
boxes1 : 预测值
boxes2 : 真实值
'''
def calc_iou(boxes1, boxes2):
    boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                       boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                       boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                       boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
    boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

    boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                       boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                       boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                       boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
    boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

    # calculate the left up point & right down point
    lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
    rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

    # intersection
    intersection = tf.maximum(0.0, rd - lu)
    inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

    # calculate the boxs1 square and boxs2 square
    square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
              (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
    square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
              (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

    union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

    return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)