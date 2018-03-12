
"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import tensorflow as tf
import keras.backend as K

def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        labels         = y_true
        classification = y_pred

        # compute the divisor: for each image in the batch, we want the number of positive anchors

        # override the -1 labels, since we treat values -1 and 0 the same way for determining the divisor
        divisor = tf.where(K.less_equal(labels, 0), K.zeros_like(labels), labels)
        divisor = K.max(divisor, axis=2, keepdims=True)
        divisor = K.cast(divisor, K.floatx())

        # compute the number of positive anchors
        divisor = K.sum(divisor, axis=1, keepdims=True)

        #  ensure we do not divide by 0
        divisor = K.maximum(1.0, divisor)

        # compute the focal loss
        alpha_factor = K.ones_like(labels) * alpha
        alpha_factor = tf.where(K.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(K.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * K.binary_crossentropy(labels, classification)

        # normalise by the number of positive anchors for each entry in the minibatch
        cls_loss = cls_loss / divisor

        # filter out "ignore" anchors
        anchor_state = K.max(labels, axis=2)  # -1 for ignore, 0 for background, 1 for object
        indices      = tf.where(K.not_equal(anchor_state, -1))

        cls_loss = tf.gather_nd(cls_loss, indices)

        # divide by the size of the minibatch
        return K.sum(cls_loss) / K.cast(K.shape(labels)[0], K.floatx())

    return _focal


def smooth_l1(sigma=3.0):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :4]
        anchor_state      = y_true[:, :, 4]

        # compute the divisor: for each image in the batch, we want the number of positive and negative anchors
        divisor = tf.where(K.not_equal(anchor_state, -1), K.ones_like(anchor_state), K.zeros_like(anchor_state))
        divisor = K.sum(divisor, axis=1, keepdims=True)
        divisor = K.maximum(1.0, divisor)

        # pad the tensor to have shape (batch_size, 1, 1) for future division
        divisor   = K.expand_dims(divisor, axis=2)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = K.abs(regression_diff)
        regression_loss = tf.where(
            K.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * K.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # normalise by the number of positive and negative anchors for each entry in the minibatch
        regression_loss = regression_loss / divisor

        # filter out "ignore" anchors
        indices         = tf.where(K.equal(anchor_state, 1))
        regression_loss = tf.gather_nd(regression_loss, indices)

        # divide by the size of the minibatch
        regression_loss = K.sum(regression_loss) / K.cast(K.shape(y_true)[0], K.floatx())

        return regression_loss

    return _smooth_l1
