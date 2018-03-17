from keras import models as KModels
from keras import layers as KLayers
from keras import initializers as KInits
from keras import optimizers as KOpts
import numpy as np
from keras_datareaders import mnistReader as reader
import pickle
from utils.progressbar.keras import ProgressBarCallback as bar


def build_generator():

    image = KLayers.Input(shape=(100,))
    label = KLayers.Input(shape=(10,))
    network = KLayers.concatenate([image, label], axis=1)
    network = KLayers.Dense(256, kernel_initializer=KInits.RandomNormal(stddev=0.02))(network)
    network = KLayers.LeakyReLU(alpha=0.2)(network)
    network = KLayers.Dense(512)(network)
    network = KLayers.LeakyReLU(alpha=0.2)(network)
    network = KLayers.Dense(1024)(network)
    network = KLayers.Dense(784, activation="tanh", kernel_initializer=KInits.RandomNormal(stddev=0.02))(network)

    # network = KLayers.Dense(128 * 7 * 7)(network)
    # network = KLayers.BatchNormalization(momentum=0.8)(network)
    # network = KLayers.LeakyReLU(alpha=0.2)(network)
    # network = KLayers.Reshape(target_shape=(7,7,128))(network)
    # network = KLayers.UpSampling2D()(network)
    #
    # network = KLayers.Conv2D(128, kernel_size=3, padding="same")(network)
    # network = KLayers.BatchNormalization(momentum=0.8)(network)
    # network = KLayers.LeakyReLU(alpha=0.2)(network)
    # network = KLayers.UpSampling2D()(network)
    #
    # network = KLayers.Conv2D(64, kernel_size=3, padding="same")(network)
    # network = KLayers.BatchNormalization(momentum=0.8)(network)
    # network = KLayers.LeakyReLU(alpha=0.2)(network)
    #
    # network = KLayers.Conv2D(1, kernel_size=3, padding="same")(network)
    # network = KLayers.LeakyReLU(alpha=0.2)(network)
    #
    # network = KLayers.Flatten()(network)
    # network = KLayers.Dense(784, activation="tanh")(network)

    network = KModels.Model(inputs=[image, label], outputs=network)
    network.compile(optimizer=adam, loss="binary_crossentropy")

    return network

def build_discriminator():
    image = KLayers.Input(shape=(784,))
    label = KLayers.Input(shape=(10,))
    network = KLayers.concatenate([image, label], axis=1)

    network = KLayers.Dense(1024, kernel_initializer=KInits.RandomNormal(stddev=0.02))(network)
    network = KLayers.LeakyReLU(alpha=0.2)(network)
    network = KLayers.Dropout(rate=0.3)(network)
    network = KLayers.Dense(512)(network)
    network = KLayers.LeakyReLU(alpha=0.2)(network)
    network = KLayers.Dropout(rate=0.3)(network)
    network = KLayers.Dense(256)(network)
    network = KLayers.LeakyReLU(alpha=0.2)(network)
    network = KLayers.Dropout(rate=0.3)(network)
    network = KLayers.Dense(1, activation="sigmoid", kernel_initializer=KInits.RandomNormal(stddev=0.02))(network)

    # network = KLayers.Dense(784)(network)
    # network = KLayers.Reshape((28,28,1))(network)
    # network = KLayers.Conv2D(32, kernel_size=3, strides=2, padding='same')(network)
    # network = KLayers.LeakyReLU(alpha=0.2)(network)
    # network = KLayers.Dropout(0.25)(network)
    # network = KLayers.Conv2D(64, kernel_size=3, strides=2, padding="same")(network)
    # network = KLayers.ZeroPadding2D(padding=((0,1),(0,1)))(network)
    # network = KLayers.LeakyReLU(alpha=0.2)(network)
    # network = KLayers.Dropout(0.25)(network)
    # network = KLayers.BatchNormalization(momentum=0.8)(network)
    # network = KLayers.Conv2D(128, kernel_size=3, strides=2, padding="same")(network)
    # network = KLayers.LeakyReLU(alpha=0.2)(network)
    # network = KLayers.Dropout(0.25)(network)
    # network = KLayers.BatchNormalization(momentum=0.8)(network)
    # network = KLayers.Conv2D(256, kernel_size=3, strides=1, padding="same")(network)
    # network = KLayers.LeakyReLU(alpha=0.2)(network)
    # network = KLayers.Dropout(0.25)(network)
    # network = KLayers.Flatten()(network)
    # network = KLayers.Dense(1, activation='sigmoid')(network)

    network = KModels.Model(inputs=[image, label], outputs=network)
    network.compile(optimizer=adam, loss="binary_crossentropy")

    return network


onehot = np.eye(10)
temp_z_ = np.random.normal(0, 1, (10, 100))
fixed_z_ = temp_z_
fixed_y_ = np.zeros((10, 1))

for i in range(9):
    fixed_z_ = np.concatenate([fixed_z_, temp_z_], 0)
    temp = np.ones((10,1)) + i
    fixed_y_ = np.concatenate([fixed_y_, temp], 0)

fixed_y_ = onehot[fixed_y_.astype(np.int32)].squeeze()
batch_size = 100
lr = 0.0002
train_epoch = 10
adam = KOpts.Adam(lr=0.0002, beta_1=0.5)
(x_train, y_train),(x_test, y_test) = reader.read_mnist()
train_set = (x_train - 0.5) / 0.5
train_label = y_train

discriminator = build_discriminator()
discriminator.trainable = False

ganInput = KLayers.Input(shape=(100,))
ganInputlabel = KLayers.Input(shape=(10,))

generator = build_generator()
x = generator([ganInput, ganInputlabel])

ganOutput = discriminator([x, ganInputlabel])
gan = KModels.Model(inputs=[ganInput, ganInputlabel], outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=adam)

progBar = bar.ProgressBarGAN(train_epoch, len(train_set) // batch_size, "D Loss:%.3f,G Loss:%.3f")
samples_image = []
for epoch in range(train_epoch):
    for iter in range(len(train_set) // batch_size):
        noise = np.random.normal(0, 1, size=[batch_size, 100])
        label = np.random.randint(0, 9, size=[batch_size, 1])
        label = onehot[label.astype(np.int32)].squeeze()
        generatedImages = generator.predict([noise, label])

        randomInt = np.random.randint(0, train_set.shape[0], size=batch_size)
        imageBatch = np.reshape(train_set[randomInt], newshape=(batch_size, 784))

        X = np.concatenate([imageBatch, generatedImages])
        Y = np.concatenate([onehot[train_label[randomInt].astype(np.int32)].squeeze(), label])

        yDis = np.zeros(2*batch_size)
        # One-sided label smoothing
        yDis[:batch_size] = 0.9

        # Train discriminator
        discriminator.trainable = True
        dloss = discriminator.train_on_batch([X, Y], yDis)

        noise = np.random.normal(0, 1, size=[batch_size, 100])
        yGen = np.ones(batch_size)
        discriminator.trainable = False
        gloss = gan.train_on_batch([noise, label], yGen)
        progBar.show(dloss, gloss)

    if epoch == 1 or epoch % 5 == 0:
        noise = np.random.normal(0, 1, size=[10, 100])
        label = np.random.randint(0, 9, size=[batch_size, 1])
        label = onehot[label.astype(np.int32)].squeeze()
        generatedImages = generator.predict([noise, label])
        generatedImages = generatedImages.reshape(10, 28, 28)
        samples_image.append(generatedImages)

with open('train_samples.pkl', 'wb') as f:
    pickle.dump(samples_image, f)
