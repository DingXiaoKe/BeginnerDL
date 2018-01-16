from keras.models import load_model
from keras_config.yoloV2Config import YoloV2Config
import colorsys
import random
import keras.backend as K
from keras_losses.yolov2 import convert_result
import tensorflow as tf
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imghdr
def filter_boxes(boxes, to, classes, to_threshold):
    """
    used to filter out boxes with confidence lower than to_threshold
    input:
    boxes: coordinates of all boxes
    to: confidence matrix
    classes: refer to result of convert_result
    to_threshold: threshold
    """
    confidence = to * classes
    #here refer to "Faster RCNN". For every point only keep points with maximum probability
    max_class = K.argmax(confidence, axis=-1)
    max_score = K.max(confidence, axis=-1)
    mask = max_score>=to_threshold #shape should be (None, 13, 13, 5)

    boxes = tf.boolean_mask(boxes, mask)
    to = tf.boolean_mask(max_score, mask)
    classes = tf.boolean_mask(max_class, mask)

    return boxes, to, classes

def draw_helper(result, image_size, max_boxes=10,to_threshold=0.6, iou_threshold=.5 ):
    """
    help to draw boxes in the image. That's to output boxes
    input:
    result: output by convert_result
    image_size:(height, width), help to calculate the relative position
    max_boxes: maximum number of boxes required by tf.non_max_suppression
    to_threshold: confidence threshold
    iou_threshold: used for non maximum supression
    """
    bxy, bwh, to, classes = result
    #convert bxy, bwh to top left and bottom right coordinates
    by_l = bxy[...,:1] - bwh[...,:1]/2.
    by_r = bxy[...,:1] + bwh[...,:1]/2.
    bx_l = bxy[...,1:] - bwh[...,1:]/2.
    bx_r = bxy[...,1:] + bwh[...,1:]/2.

    boxes = K.concatenate([bx_l,by_l, bx_r, by_r])
    #drop boxes with confidence lower than to_threshold

    boxes , to, classes = filter_boxes(boxes, to, classes, to_threshold)

    #scale back to image size
    height = image_size[0]
    width = image_size[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    #get use non-maximum-supression
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    nms_index = tf.image.non_max_suppression(
        boxes, to, max_boxes_tensor, iou_threshold=iou_threshold)
    boxes = K.gather(boxes, nms_index)
    to = K.gather(to, nms_index)
    classes = K.gather(classes, nms_index)
    return boxes, to, classes

cfg = YoloV2Config()

MODEL_PATH = "../weights/model_yolov2_416_coco.h5"
OUTPUT_PATH = "../predictImages/yolov2_out"
TEST_PATH = "../predictImages"
CLASSES_PATH = 'coco_classes.txt'
if not os.path.exists(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)

sess = K.get_session()

with open(CLASSES_PATH) as f:
    class_names = f.readlines()
class_names = [c.strip() for c in class_names]
anchors = cfg.ANCHOR_VALUE
yolo_model = load_model(MODEL_PATH)

# Verify model, anchors, and classes are compatible
num_classes = len(class_names)
num_anchors = len(anchors)
# TODO: Assumes dim ordering is channel last
model_output_channels = yolo_model.layers[-1].output_shape[-1]
assert model_output_channels == num_anchors * (num_classes + 5), \
    'Mismatch between model and given anchor and class sizes. ' \
    'Specify matching anchors and classes with --anchors_path and ' \
    '--classes_path flags.'
print('{} model, anchors, and classes loaded.'.format(MODEL_PATH))

# Check if model is fully convolutional, assuming channel last order.
model_image_size = yolo_model.layers[0].input_shape[1:3]
is_fixed_size = model_image_size != (None, None)

# Generate colors for drawing bounding boxes.
hsv_tuples = [(x / float(len(class_names)), 1., 1.)
              for x in range(len(class_names))]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))
random.seed(10101)  # Fixed seed for consistent colors across runs.
random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
random.seed(None)  # Reset seed to default.

# Generate output tensor targets for filtered bounding boxes.
# TODO: Wrap these backend operations with Keras layers.
yolo_outputs = convert_result(yolo_model.output, anchors, len(class_names))
input_image_shape = K.placeholder(shape=(2, ))
boxes, scores, classes = draw_helper(
    yolo_outputs,
    input_image_shape,
    to_threshold=.4,
    iou_threshold=.5)

for image_file in os.listdir(TEST_PATH):
    try:
        image_type = imghdr.what(os.path.join(TEST_PATH, image_file))
        if not image_type:
            continue
    except Exception as e:
        continue
    image = Image.open(os.path.join(TEST_PATH, image_file))
    if is_fixed_size:  # TODO: When resizing we can use minibatch input.
        resized_image = image.resize(
            tuple(reversed(model_image_size)), Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
    else:
        # Due to skip connection + max pooling in YOLO_v2, inputs must have
        # width and height as multiples of 32.
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        resized_image = image.resize(new_image_size, Image.BICUBIC)
        image_data = np.array(resized_image, dtype='float32')
        print(image_data.shape)

    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            yolo_model.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))

    font = ImageFont.truetype(
        font='../common/FiraMono-Medium.otf',
        size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 300

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        #the result's origin is in top left
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    image.save(os.path.join(OUTPUT_PATH, image_file), quality=90)
sess.close()