
from lib.config.mnist import MNISTConfig
from lib.datareader.common import read_mnist
from lib.utils.progressbar.ProgressBar import ProgressBar

from keras.utils import multi_gpu_model
from keras import models as KModels
from keras import layers as KLayers
from keras import optimizers as KOpts
import numpy as np
from matplotlib import pyplot as plt

PHRASE = "TEST"
GPU_NUMS = 2
batchSize = 100
epochs = 30
cfg = MNISTConfig()

if PHRASE == "TRAIN":
    np.random.seed(1000)
    randomDim = 100
    (X_train, y_train), (X_test, y_test) = read_mnist()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(60000, 784)

    adam = KOpts.Adam(lr=0.0002, beta_1=0.5)

    def build_generator(randomDim):
        img = KLayers.Input(shape=(randomDim,))
        network = KLayers.Dense(units=128 * 7 * 7)(img)
        network = KLayers.Activation('relu')(network)
        network = KLayers.Reshape((7,7,128))(network)
        network = KLayers.BatchNormalization(momentum=0.8)(network)
        network = KLayers.UpSampling2D()(network)

        network = KLayers.Conv2D(128, kernel_size=3, padding='same')(network)
        network = KLayers.Activation('relu')(network)
        network = KLayers.BatchNormalization(momentum=0.8)(network)
        network = KLayers.UpSampling2D()(network)

        network = KLayers.Conv2D(64, kernel_size=3, padding='same')(network)
        network = KLayers.Activation('relu')(network)
        network = KLayers.BatchNormalization(momentum=0.8)(network)

        network = KLayers.Conv2D(1, kernel_size=3, padding='same')(network)
        network = KLayers.Activation("tanh")(network)

        model = KModels.Model(inputs=img, outputs=network)
        # model = KModels.Sequential()
        # model.add(KLayers.Dense(128 * 7 * 7, input_dim=randomDim)) # ?,100 => ?, 128*7*7
        # model.add(KLayers.Activation('relu'))
        # model.add(KLayers.Reshape((7,7,128))) # ?, 128*7*7 => ?,7,7,128
        # model.add(KLayers.BatchNormalization(momentum=0.8))
        # model.add(KLayers.UpSampling2D()) # ?,7,7,128 => ?,14,14,128

        # model.add(KLayers.Conv2D(128, kernel_size=3, padding='same')) # ?,14,14,128 => ?,14,14,128
        # model.add(KLayers.Activation('relu'))
        # model.add(KLayers.BatchNormalization(momentum=0.8))
        # model.add(KLayers.UpSampling2D())  # ?,14,14,128 => ?,28,28,128

        # model.add(KLayers.Conv2D(64, kernel_size=3, padding='same')) # ?,28,28,128 => ?,28,28,64
        # model.add(KLayers.Activation('relu'))
        # model.add(KLayers.BatchNormalization(momentum=0.8))
        #
        # model.add(KLayers.Conv2D(1, kernel_size=3, padding='same')) # ?,28,28,64 => ?,28,28,1
        # model.add(KLayers.Activation("tanh"))
        # if GPU_NUMS > 1:
        #     model = multi_gpu_model(model, GPU_NUMS)
        model.compile(loss="binary_crossentropy", optimizer=adam)

        return model

    def build_discriminator():
        img = KLayers.Input(shape=(28,28,1))
        network = KLayers.Conv2D(32, kernel_size=3, strides=2,  padding='same')(img)
        network = KLayers.Activation('relu')(network)
        network = KLayers.Dropout(0.25)(network)
        network = KLayers.Conv2D(64, kernel_size=3, strides=2, padding="same")(network)
        network = KLayers.ZeroPadding2D(padding=((0,1),(0,1)))(network)
        network = KLayers.Activation('relu')(network)
        network = KLayers.Dropout(0.25)(network)
        network = KLayers.BatchNormalization(momentum=0.8)(network)
        network = KLayers.Conv2D(128, kernel_size=3, strides=2, padding="same")(network)
        network = KLayers.Activation('relu')(network)
        network = KLayers.Dropout(0.25)(network)
        network = KLayers.BatchNormalization(momentum=0.8)(network)
        network = KLayers.Conv2D(256, kernel_size=3, strides=1, padding="same")(network)
        network = KLayers.Activation('relu')(network)
        network = KLayers.Dropout(0.25)(network)
        network = KLayers.Flatten()(network)
        network = KLayers.Dense(1, activation='sigmoid')(network)
        model = KModels.Model(inputs=img, outputs=network)
        # model = KModels.Sequential()
        # model.add(KLayers.Conv2D(32, kernel_size=3, strides=2, input_shape=(28,28,1), padding='same')) #?,28,28,1 => ?,14,14,32
        # model.add(KLayers.Activation('relu'))
        # model.add(KLayers.Dropout(0.25))
        # model.add(KLayers.Conv2D(64, kernel_size=3, strides=2, padding="same")) # => ?,7,7,64
        # model.add(KLayers.ZeroPadding2D(padding=((0,1),(0,1)))) # => ?,8,8,64
        # model.add(KLayers.Activation('relu'))
        # model.add(KLayers.Dropout(0.25))
        # model.add(KLayers.BatchNormalization(momentum=0.8))
        # model.add(KLayers.Conv2D(128, kernel_size=3, strides=2, padding="same")) # => ?,4,4,128
        # model.add(KLayers.Activation('relu'))
        # model.add(KLayers.Dropout(0.25))
        # model.add(KLayers.BatchNormalization(momentum=0.8))
        # model.add(KLayers.Conv2D(256, kernel_size=3, strides=1, padding="same")) #=> ?,4,4,256
        # model.add(KLayers.Activation('relu'))
        # model.add(KLayers.Dropout(0.25))
        #
        # model.add(KLayers.Flatten())
        # model.add(KLayers.Dense(1, activation='sigmoid'))
        if GPU_NUMS > 1:
            model = multi_gpu_model(model, GPU_NUMS)

        model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

        return model

    discriminator = build_discriminator()
    discriminator.trainable = False

    ganInput = KLayers.Input(shape=(randomDim,))
    generator = build_generator(randomDim)
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = KModels.Model(inputs=ganInput, outputs=ganOutput)
    if GPU_NUMS > 1:
        model = multi_gpu_model(gan, GPU_NUMS)
        # gan = kutils.multi_gpu_model(gan, 2)
    gan.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    dLosses = []
    gLosses = []


    batchCount = X_train.shape[0] // batchSize
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)
    progBar = ProgressBar(epochs, batchCount, "D Loss:%.3f,D Acc:%.3f;G Loss:%.3f,G Acc:%.3f")
    samples_image = []
    for e in range(1, (epochs+1)):
        # Get a random set of input noise and images

        for _ in range(batchCount):
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

            imageBatch = np.reshape(imageBatch, newshape=(batchSize, 28,28,1))

            # Generate fake MNIST images
            generatedImages = generator.predict(noise)
            # print np.shape(imageBatch), np.shape(generatedImages)
            X = np.concatenate([imageBatch, generatedImages])

            # Labels for generated and real data
            yDis = np.zeros(2*batchSize)
            # One-sided label smoothing
            yDis[:batchSize] = 0.9

            # Train discriminator
            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = gan.train_on_batch(noise, yGen)
            progBar.show(dloss[0], dloss[1], gloss[0],gloss[1])

        dLosses.append(dloss)
        gLosses.append(gloss)
        if e == 1 or e % 5 == 0:
            noise = np.random.normal(0, 1, size=[100, randomDim])
            generatedImages = generator.predict(noise)
            generatedImages = generatedImages.reshape(100, 28, 28)
            samples_image.append(generatedImages)

    generator.save("mnist_generator.h5")
else:
    generator = KModels.load_model("mnist_generator.h5")
    noise = np.random.normal(0, 1, size=[100, 100])
    generated_images = generator.predict(noise)

    plt.figure(num='astronaut',figsize=(8,8))

    for index, img in enumerate(generated_images):
        plt.subplot(10,10,index+1)
        plt.imshow(img.reshape(28,28), plt.cm.gray)

    plt.show()




