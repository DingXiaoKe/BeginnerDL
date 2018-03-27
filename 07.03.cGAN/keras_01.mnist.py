# coding=utf-8
import numpy as np

from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras.layers import Input, Dense,LeakyReLU,BatchNormalization, Reshape, Embedding, Flatten, multiply, Dropout
from keras.models import Sequential, Model

from lib.datareader.common import read_mnist
from lib.utils.progressbar.ProgressBar import ProgressBar
from lib.config.mnist import MNISTConfig

config = MNISTConfig()
EPOCH = 10000
GPU_NUMS = 1
BATCH_SIZE = config.BATCH_SIZE
NOISE_DIM = 100

'''
定义Generator网络, 输入为NOISE_DIM,输出为(28,28,1)
'''
def builder_generator():
    image_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNEL)
    model = Sequential()
    model.add(Dense(units=256, input_dim=NOISE_DIM))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(units=1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(units=np.prod(image_shape), activation='tanh'))
    model.add(Reshape(image_shape))

    noise = Input(shape=(NOISE_DIM,))
    label = Input(shape=(1,), dtype='int32')

    embed = Embedding(config.NUM_OUTPUTS, NOISE_DIM)(label)
    label_embedding = Flatten()(embed)

    model_input = multiply([noise, label_embedding])

    model_output = model(model_input)

    return Model([noise, label], model_output)

'''
定义Discriminator网络
'''
def builder_discriminator():
    image_shape = (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNEL)
    model = Sequential()

    model.add(Dense(units=512, input_dim=(np.prod(image_shape))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.4))
    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.4))

    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=image_shape)
    label = Input(shape=(1,), dtype='int32')

    flatten_label = Flatten()(Embedding(config.NUM_OUTPUTS, np.prod(image_shape))(label))
    flatten_img = Flatten()(img)

    model_input = multiply([flatten_img, flatten_label])

    model_output = model(model_input)

    return Model([img, label], model_output)

def save_images(generator):
    noise = np.random.normal(0, 1, (2 * 5, NOISE_DIM))
    label = np.arange(0, 10).reshape(-1, 1)

    generate_image = generator.predict([noise, label])

    gen_imgs = 0.5 * generate_image + 0.5

    fig, axs = plt.subplots(2, 5)
    fig.suptitle("CGAN: Generated digits", fontsize=12)
    cnt = 0
    for i in range(2):
        for j in range(5):
            axs[i,j].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
            axs[i,j].set_title("Digit: %d" % label[cnt])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("output/%d.png" % epoch)
    plt.close()

'''
开始训练
'''
'''
1. 读入数据,然后归一化到(-1,1)
'''
(x_train, y_train), (_,_) = read_mnist("../ganData/mnist.npz")
x_train = (x_train.astype(np.float32) - 127.5) / 127.5 # 127.5 = 255 / 2
x_train = np.expand_dims(x_train, axis=3) # shape从(60000,28,28) => (60000,28,28,1)
y_train = np.reshape(y_train, newshape=(-1,1))

'''
2. 生成网络
'''
Generator = builder_generator()
Discriminator = builder_discriminator()
optimizer = Adam(0.0002, 0.5)
Generator.compile(loss='binary_crossentropy', optimizer=optimizer)
Discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
noise_temp = Input(shape=(NOISE_DIM,))
label_temp = Input(shape=(1,))
img = Generator([noise_temp, label_temp])
Discriminator.trainable = False

valid = Discriminator([img, label_temp])
Gan = Model([noise_temp, label_temp], valid)
Gan.compile(loss='binary_crossentropy', optimizer=optimizer)

'''
开始训练
'''
half_batch = int(BATCH_SIZE / 2)
proBar = ProgressBar(1, EPOCH, "D loss: %f, acc.: %.2f%%; G loss: %f")
for epoch in range(1, EPOCH + 1):
    idx = np.random.randint(0, x_train.shape[0], size=half_batch)
    image, label = x_train[idx], y_train[idx]

    noise = np.random.normal(0, 1, (half_batch, 100))
    generate_image = Generator.predict([noise, label])

    valid = np.ones((half_batch, 1))
    fake = np.zeros((half_batch, 1))

    d_loss_real = Discriminator.train_on_batch([image, label], valid)
    d_loss_fake = Discriminator.train_on_batch([generate_image, label], fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.normal(0, 1, (BATCH_SIZE, 100))

    valid = np.ones((BATCH_SIZE, 1))

    sampled_labels = np.random.randint(0, 10, BATCH_SIZE).reshape(-1, 1)

    # Train the generator
    g_loss = Gan.train_on_batch([noise, sampled_labels], valid)
    proBar.show(d_loss[0], 100*d_loss[1], g_loss)
    if epoch % 100 == 0:
        save_images(Generator)

Generator.save("output/keras_mnist_generator.h5")