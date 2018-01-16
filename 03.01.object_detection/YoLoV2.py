from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Lambda, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
import keras.backend as K
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras.models import Model
from keras_losses.yolov2 import loss_function
from keras.optimizers import SGD
from keras_datareaders.yolov2_voc_reader import data_generator
from keras_callbacks.ProgressBarCallback import ProgressBarCallback
from keras_config.yoloV2Config import YoloV2Config
from keras.preprocessing.image import ImageDataGenerator
cfg = YoloV2Config()

def model_out(images):
    L2 = cfg.L2
    ALPHA = cfg.LEAKEY_ALPHA
    network = Conv2D(filters=32,kernel_size=(3,3), kernel_initializer='he_normal', padding="same",use_bias=False, kernel_regularizer=l2(L2))(images)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding="same")(network)

    network = Conv2D(filters=64, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = MaxPooling2D(pool_size=2, strides=2, padding="same")(network)

    network = Conv2D(filters=128, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = Conv2D(filters=64, kernel_size=1, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = Conv2D(filters=128, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = MaxPooling2D(pool_size=2, strides=2, padding="same")(network)

    network = Conv2D(filters=256, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = Conv2D(filters=128, kernel_size=1, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = Conv2D(filters=256, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)
    network = MaxPooling2D(pool_size=2, strides=2, padding="same")(network)

    network = Conv2D(filters=512, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = Conv2D(filters=256, kernel_size=1, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = Conv2D(filters=512, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = Conv2D(filters=256, kernel_size=1, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = Conv2D(filters=512, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    out1 = LeakyReLU(alpha=ALPHA)(network)
    network = MaxPooling2D(pool_size=2, strides=2, padding="same")(out1)

    network = Conv2D(filters=1024, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = Conv2D(filters=512, kernel_size=1, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = Conv2D(filters=1024, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = Conv2D(filters=512, kernel_size=1, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = Conv2D(filters=1024, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = Conv2D(filters=1024, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = Conv2D(filters=1024, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network2 = Conv2D(filters=64, kernel_size=1, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(out1)
    network2 = BatchNormalization()(network2)
    network2 = LeakyReLU(alpha=ALPHA)(network2)

    network2 = Lambda(lambda x : tf.space_to_depth(x, block_size=2),
                      output_shape=(cfg.BOUX_CELL_SIZE, cfg.BOUX_CELL_SIZE, 256))(network2)
    network = Concatenate()([network2, network])

    network = Conv2D(filters=1024, kernel_size=3, kernel_initializer='he_normal', kernel_regularizer=l2(L2), use_bias=False,padding="same")(network)
    network = BatchNormalization()(network)
    network = LeakyReLU(alpha=ALPHA)(network)

    network = Conv2D(filters=cfg.NUM_OUTPUTS, kernel_size=1, kernel_initializer='he_normal', use_bias=True, kernel_regularizer=l2(L2), padding="same")(network)

    return network

session = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
K.set_session(session)
image_input = Input(shape=(cfg.IMAGE_SIZE,cfg.IMAGE_SIZE,cfg.IMAGE_CHANNEL))
y_pred = model_out(image_input)

y1 = Input(shape=(5,))
y2 = Input(shape=(cfg.IMAGE_SIZE//cfg.BOX_CELLS_NUM, cfg.IMAGE_SIZE//cfg.BOX_CELLS_NUM, cfg.ANCHOR_LENGTH, 1))
y3 = Input(shape=(cfg.IMAGE_SIZE//cfg.BOX_CELLS_NUM, cfg.IMAGE_SIZE//cfg.BOX_CELLS_NUM, cfg.ANCHOR_LENGTH, 5))
loss_out = Lambda(loss_function, output_shape=(1,))([y_pred, y1, y2, y3])
model = Model(inputs=[image_input, y1, y2, y3], outputs=[loss_out])

if cfg.GPU_NUM >= 2:
    model = multi_gpu_model(model, gpus=cfg.GPU_NUM)

model.compile(loss=lambda y_true, y_pred : y_pred,
              optimizer=SGD(lr=0.00001, momentum=0.9, nesterov=True),
              metrics=['acc'])

train_image_data_generator = ImageDataGenerator(
    horizontal_flip=True,
)

reader = data_generator("../data/VOC2007/train.txt",
                        batch_size=cfg.BATCH_SIZE, target_image_size=(cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
probar = ProgressBarCallback()
model.fit_generator(generator=reader,steps_per_epoch=5011/cfg.BATCH_SIZE,
                    epochs=cfg.EPOCH_NUM, verbose=0,callbacks=[probar])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("yolo_v2.h5")
print("Saved model to disk")

