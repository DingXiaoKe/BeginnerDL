import keras.backend as K
import tensorflow as tf
from keras.losses import categorical_crossentropy
lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4

def rpn_loss_cls(num_anchors):
    def rpn_loss_cls_fixed_num(y_true, y_pred):
        Kbin = K.binary_crossentropy(y_pred[:,:,:,:], y_true[:,:,:,num_anchors:])
        Ksum1 = K.sum(y_true[:,:,:,:num_anchors] * Kbin)
        Ksum2 = K.sum(epsilon + y_true[:,:,:,:num_anchors])
        return lambda_rpn_class * Ksum1 / Ksum2
    return rpn_loss_cls_fixed_num

def rpn_loss_regr(num_anchors):
    def rpn_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:,:,:,4 * num_anchors:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
        Ksum1 = K.sum(y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5)))
        Ksum2 =  K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])
        return lambda_rpn_regr * Ksum1 / Ksum2
    return rpn_loss_regr_fixed_num

def class_loss_cls(y_true, y_pred):
    return lambda_cls_class * K.mean(categorical_crossentropy(y_true[0,:,:], y_pred[0,:,:]))

def class_loss_regr(num_classes):
    def class_loss_regr_fixed_num(y_true, y_pred):
        x = y_true[:,:, : 4 * num_classes:] - y_pred
        x_abs = K.abs(x)
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)
        Ksum1 = K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5)))
        Ksum2 = K.sum(epsilon + y_true[:, :, :4*num_classes])
        return lambda_cls_regr * Ksum1 / Ksum2
    return class_loss_regr_fixed_num