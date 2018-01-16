import tensorflow as tf
import numpy as np
import keras.backend as K
from keras_config.yoloV2Config import YoloV2Config

cfg = YoloV2Config()

def loss_function(args):
    #should work fun
    """
    y_true (batch, 13, 13, 425) tensor
    y1 (batch, 5) tensor
    y2 (batch, 13,13,5,1)
    y3 (batch, 13,13,5,5)
    """
    y_pred, y1, y2, y3 = args
    #converted_result = convert_result(y_pred, anchors, nb_classes)
    return loss_calculator(y_pred, y2, y3)

def loss_calculator(output, object_mask, object_value):
    """
    calculate loss on the basis of a batch
    para:
        output: output by the net. (13, 13, 485)
        anchors: list of anchor info. value is correspoding length. (that's value*32 convert
            back to absolute pixel values)
            ex:
                0.7684, 0.9980
                1.3340, 3.1890
                .....
        object_mask: shape(batch_size, 13, 13, 5, 1), with entry equals 1 means this anchor is the
            right one. 1obj in loss equation
        object_value: shape(batch_size, 13, 13, 5, 5), indicates the x, y, w, h and class for the
            right box
    """

    #use convert_result to convert output. bxy is bx, by.
    bxy, bwh, to, classes = convert_result(output, cfg.ANCHOR_VALUE, cfg.NUM_CLASSES)

    #leave the ratio unassigned right now
    alpha1 = 5.0
    alpha2 = 5.0
    alpha3 = 1.0
    alpha4 = 0.5
    alpha5 = 1.0

    #first term coordinate_loss
    bxy_sigmoid = bxy - tf.floor(bxy)
    bxy_loss = K.sum(K.square(bxy_sigmoid - object_value[...,0:2])*object_mask)

    #second term
    bwh_loss = K.sum(K.square(K.sqrt(bwh)-K.sqrt(object_value[...,2:4]))*object_mask)

    #third term
    to_obj_loss = K.sum(K.square(1-to)*object_mask)

    #forth term. TODO, need to multiply another factor.  (1 - object_detection)
    to_noobj_loss = K.sum(K.square(0-to)*(1-object_mask))

    #fifth term
    onehot_class = K.one_hot(tf.to_int32(object_value[...,4]), cfg.NUM_CLASSES)
    class_loss = K.sum(K.square(onehot_class-classes)*object_mask)

    #total loss
    result = alpha1*bxy_loss + alpha2*bwh_loss + alpha3*to_obj_loss + \
             alpha4*to_noobj_loss + alpha5*class_loss

    return result

def convert_result(output, anchors, nb_classes):
    """
    convert the model output into train label or test result comparable format.
    input:
    output: the output of model, for example, with shape, (batch_size, 13, 13, 425)
    anchors: the precomputed anchor size, list of tuples, every tuple indicates one width and height
    nb_classes: int, the number of prediction classes, to verify data integrity
    """
    anchors_length = K.shape(anchors)[0]
    output_shape = K.shape(output)

    tf_anchors = K.reshape(K.variable(anchors), [1, 1, 1, anchors_length, 2])

    #represent cx, cy
    height_index = K.arange(0,stop=output_shape[1])
    width_index = K.arange(0,stop=output_shape[2])
    tmp1, tmp2 = tf.meshgrid(height_index, width_index)
    conv_index = tf.reshape(tf.concat([tmp1, tmp2], axis=0),(2, output_shape[1],output_shape[2]))
    conv_index = tf.transpose(conv_index, (1,2,0))
    conv_index = K.expand_dims(K.expand_dims(conv_index, 0),-2)#shape will be (1, 13, 13, 1, 2)
    conv_index = K.cast(conv_index, K.dtype(output))

    #reshape output
    output = K.reshape(output, [-1, output_shape[1], output_shape[2], anchors_length, nb_classes+5])

    #get sigmoid tx, ty, tw, th ,to in the paper
    bxy = K.sigmoid(output[...,:2])
    bwh = K.exp(output[...,2:4])
    to = K.sigmoid(output[...,4:5])
    classes = K.softmax(output[...,5:])

    #use the equation to recover x,y, and get ratio
    dims = K.cast(K.reshape(output_shape[1:3],(1,1,1,1,2)), K.dtype(output))
    bxy = (bxy + conv_index)/dims
    bwh = bwh*tf_anchors/dims

    #the returned shape is (None, 13, 13, 5, 2), (None, 13, 13, 5, 2), (None, 13, 13, 5, 1), (None, 13, 13, 5, nb_clases), 5 is anchor length
    return bxy, bwh, to, classes