from keras_config import mnist as config
from keras_datareaders import mnistReader as reader
from keras_callbacks import ProgressBarCallback as bar

from keras import models as KModels
from keras import layers as KLayers
from keras import initializers as KInits
from keras import optimizers as KOpts
import numpy as np
import pickle
from matplotlib import pyplot as plt

PHRASE = "TRAIN"

cfg = config.MNISTConfig()

if PHRASE == "TRAIN":
    np.random.seed(1000)
    randomDim = 100
    (X_train, y_train), (X_test, y_test) = reader.read_mnist("../data/mnist.npz")
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(60000, 784)

    adam = KOpts.Adam(lr=0.0002, beta_1=0.5)

    def build_generator(randomDim):
        model = KModels.Sequential()
        model.add(KLayers.Dense(128 * 7 * 7, input_dim=randomDim)) # ?,100 => ?, 128*7*7
        model.add(KLayers.LeakyReLU(alpha=0.2))
        model.add(KLayers.Reshape((7,7,128))) # ?, 128*7*7 => ?,7,7,128
        model.add(KLayers.BatchNormalization(momentum=0.8))
        model.add(KLayers.UpSampling2D()) # ?,7,7,128 => ?,14,14,128

        model.add(KLayers.Conv2D(128, kernel_size=3, padding='same')) # ?,14,14,128 => ?,14,14,128
        model.add(KLayers.LeakyReLU(alpha=0.2))
        model.add(KLayers.BatchNormalization(momentum=0.8))
        model.add(KLayers.UpSampling2D())  # ?,14,14,128 => ?,28,28,128

        model.add(KLayers.Conv2D(64, kernel_size=3, padding='same')) # ?,28,28,128 => ?,28,28,64
        model.add(KLayers.LeakyReLU(alpha=0.2))
        model.add(KLayers.BatchNormalization(momentum=0.8))

        model.add(KLayers.Conv2D(1, kernel_size=3, padding='same')) # ?,28,28,64 => ?,28,28,1
        model.add(KLayers.Activation("tanh"))

        model.compile(loss="binary_crossentropy", optimizer=adam)

        return model

    def build_discriminator():
        model = KModels.Sequential()

        model.add(KLayers.Conv2D(32, kernel_size=3, strides=2, input_shape=(28,28,1), padding='same')) #?,28,28,1 => ?,14,14,32
        model.add(KLayers.LeakyReLU(alpha=0.2))
        model.add(KLayers.Dropout(0.25))
        model.add(KLayers.Conv2D(64, kernel_size=3, strides=2, padding="same")) # => ?,7,7,64
        model.add(KLayers.ZeroPadding2D(padding=((0,1),(0,1)))) # => ?,8,8,64
        model.add(KLayers.LeakyReLU(alpha=0.2))
        model.add(KLayers.Dropout(0.25))
        model.add(KLayers.BatchNormalization(momentum=0.8))
        model.add(KLayers.Conv2D(128, kernel_size=3, strides=2, padding="same")) # => ?,4,4,128
        model.add(KLayers.LeakyReLU(alpha=0.2))
        model.add(KLayers.Dropout(0.25))
        model.add(KLayers.BatchNormalization(momentum=0.8))
        model.add(KLayers.Conv2D(256, kernel_size=3, strides=1, padding="same")) #=> ?,4,4,256
        model.add(KLayers.LeakyReLU(alpha=0.2))
        model.add(KLayers.Dropout(0.25))

        model.add(KLayers.Flatten())
        model.add(KLayers.Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=adam)

        return model

    discriminator = build_discriminator()
    discriminator.trainable = False

    ganInput = KLayers.Input(shape=(randomDim,))
    generator = build_generator(randomDim)
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = KModels.Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=adam)

    dLosses = []
    gLosses = []
    batchSize = 32
    epochs = 20

    batchCount = X_train.shape[0] // batchSize
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)
    progBar = bar.ProgressBarGAN(epochs, batchCount, "D Loss:%.3f,G Loss:%.3f")
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
            progBar.show(dloss, gloss)

        dLosses.append(dloss)
        gLosses.append(gloss)
        if e == 1 or e % 5 == 0:
            noise = np.random.normal(0, 1, size=[100, randomDim])
            generatedImages = generator.predict(noise)
            generatedImages = generatedImages.reshape(100, 28, 28)
            samples_image.append(generatedImages)

    with open('train_samples.pkl', 'wb') as f:
        pickle.dump(samples_image, f)
else:
    with open('train_samples.pkl', 'rb') as f:
        samples = pickle.load(f)
    # view_samples(-1, samples)

    show_imgs = samples

    # 指定图片形状
    rows, cols = len(show_imgs), 10
    fig, axes = plt.subplots(figsize=(30,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

    for sample, ax_row in zip(show_imgs, axes):
        for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
            ax.imshow(img.reshape((28,28)), cmap='Greys_r')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    plt.show()
