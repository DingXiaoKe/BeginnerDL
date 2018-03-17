import numpy as np
import pickle
from matplotlib import pyplot as plt

from keras.models import Model, Sequential
from keras.layers import Dense, LeakyReLU, Dropout, Input
from keras import initializers as KInits
from keras import optimizers as KOpts

from lib.config.mnist import MNISTConfig
from lib.datareader.common import read_mnist
from lib.utils.progressbar.keras.ProgressBarCallback import ProgressBar

PHRASE = "TRAIN"

cfg = MNISTConfig()

if PHRASE == "TRAIN":
    np.random.seed(1000)
    randomDim = 100
    (X_train, y_train), (X_test, y_test) = read_mnist("../data/mnist.npz")
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(60000, 784)

    adam = KOpts.Adam(lr=0.0002, beta_1=0.5)

    def build_generator(randomDim):
        model = Sequential()
        model.add(Dense(256, input_dim=randomDim, kernel_initializer=KInits.RandomNormal(stddev=0.02)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(784, activation="tanh"))
        model.compile(loss="binary_crossentropy", optimizer=adam)

        return model

    def build_discriminator():
        model = Sequential()

        model.add(Dense(1024, input_dim=784, kernel_initializer=KInits.RandomNormal(stddev=0.02)))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        model.add(Dense(512))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        model.add(Dense(256))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=adam)

        return model

    discriminator = build_discriminator()
    discriminator.trainable = False

    ganInput = Input(shape=(randomDim,))
    generator = build_generator(randomDim)
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=adam)

    dLosses = []
    gLosses = []
    batchSize = 64
    epochs = 20

    batchCount = X_train.shape[0] // batchSize
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)
    progBar = ProgressBar(epochs, batchCount, "D Loss:%.3f,G Loss:%.3f")
    samples_image = []
    for e in range(1, (epochs+1)):
        # Get a random set of input noise and images

        for _ in range(batchCount):
            noise = np.random.normal(0, 1, size=[batchSize, randomDim])
            imageBatch = X_train[np.random.randint(0, X_train.shape[0], size=batchSize)]

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
