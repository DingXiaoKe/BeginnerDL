from keras import layers as KLayers
from keras import models as KModels
from keras import optimizers as KOpts
from keras_datareaders import ClassificationReader as reader
from utils.progressbar.keras import ProgressBarCallback as bar
import numpy as np
import pickle
from matplotlib import pyplot as plt
from keras import utils as kutils

def builder_generator(randomDim):
    model = KModels.Sequential()

    model.add(KLayers.Dense(512*4*4,input_dim=randomDim)) # ?,100 => ?,512*4*4
    model.add(KLayers.Activation("relu"))
    model.add(KLayers.Reshape((4,4,512)))
    model.add(KLayers.BatchNormalization(momentum=0.8))
    model.add(KLayers.Dropout(rate=0.8))

    model.add(KLayers.UpSampling2D()) # ?,4,4,512 => ?,8,8,512

    model.add(KLayers.Conv2D(256, kernel_size=3, padding='same')) #?,8,8,512 => ?,8,8,256
    model.add(KLayers.Activation("relu"))
    model.add(KLayers.BatchNormalization(momentum=0.8))
    model.add(KLayers.Dropout(rate=0.8))

    model.add(KLayers.UpSampling2D())  #?,8,8,256 => ?,16,16,256

    model.add((KLayers.Conv2D(128, kernel_size=3, padding='same'))) #?,16,16,256 => ?,16,16,128
    model.add(KLayers.Activation("relu"))
    model.add(KLayers.BatchNormalization(momentum=0.8))
    model.add(KLayers.Dropout(rate=0.8))

    model.add(KLayers.UpSampling2D()) #?,16,16,128 => ?,32,32,128

    model.add(KLayers.Conv2D(3, kernel_size=3, padding='same')) #?,32,32,128 => ?,32,32,3
    model.add(KLayers.Activation("tanh"))

    return model

def builder_discriminator():
    model = KModels.Sequential()

    model.add(KLayers.Conv2D(128, kernel_size=3, strides=2, padding='same', input_shape=(32,32,3))) #?,32,32,3 => ?,16,16,128
    model.add(KLayers.LeakyReLU(alpha=0.2))
    model.add(KLayers.Dropout(rate=0.8))

    model.add(KLayers.Conv2D(256, kernel_size=3, strides=2, padding='same')) #?,16,16,128 => ?,8,8,256
    model.add(KLayers.LeakyReLU(alpha=0.2))
    model.add(KLayers.BatchNormalization(momentum=0.8))
    model.add(KLayers.Dropout(rate=0.8))

    model.add(KLayers.Conv2D(512, kernel_size=3, strides=2, padding='same')) #?,8,8,256 => ?,4,4,512
    model.add(KLayers.LeakyReLU(alpha=0.2))
    model.add(KLayers.BatchNormalization(momentum=0.8))
    model.add(KLayers.Dropout(rate=0.8))

    model.add(KLayers.Flatten())
    model.add(KLayers.Dense(1, activation="sigmoid"))

    return model

PHRASE = "TRAIN"

if PHRASE == "TRAIN":
    adam = KOpts.Adam(lr=0.0002, beta_1=0.5)
    reader = reader.ClassificationReader()
    imageList, labelList = reader.readData()

    discriminator = builder_discriminator()
    discriminator.compile(optimizer=adam, loss='binary_crossentropy')
    discriminator.trainable = False

    generator = builder_generator(100)
    ganInput = KLayers.Input(shape=(100,))
    ganOutput = discriminator(generator(ganInput))
    dcgan = KModels.Model(inputs=ganInput, outputs=ganOutput)
    dcgan = kutils.multi_gpu_model(dcgan, 2)
    dcgan.compile(loss='binary_crossentropy', optimizer=adam)
    dLosses = []
    gLosses = []
    batchSize = 32
    epochs = 20
    dloss = 0
    gloss = 0
    for epoch in range(1, (epochs + 1)):
        batchCount = imageList.shape[0] // batchSize
        print('Epochs:', epochs)
        print('Batch size:', batchSize)
        print('Batches per epoch:', batchCount)
        progBar = bar.ProgressBarGAN(epochs, batchCount, "D Loss:%.3f,G Loss:%.3f")
        samples_image = []

        for _ in range(batchCount):
            noise = np.random.normal(0, 1, size=[batchSize, 100])
            imageBatch = imageList[np.random.randint(0, imageList.shape[0], size=batchSize)]

            # Generate fake cifar images
            generatedImages = generator.predict(noise)
            # print np.shape(imageBatch), np.shape(generatedImages)
            discriminator.trainable = True

            d_loss_real = discriminator.train_on_batch(imageBatch, np.ones(batchSize))
            d_loss_fake = discriminator.train_on_batch(generatedImages, np.zeros(batchSize))
            dloss = np.add(d_loss_fake,d_loss_real) * 0.5
            # dloss = discriminator.train_on_batch(X, yDis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batchSize, 100])
            yGen = np.ones(batchSize)
            discriminator.trainable = False
            gloss = dcgan.train_on_batch(noise, yGen)
            progBar.show(dloss, gloss)

        dLosses.append(dloss)
        gLosses.append(gloss)
        if epoch == 1 or epoch % 5 == 0:
            noise = np.random.normal(0, 1, size=[10, 100])
            generatedImages = generator.predict(noise)
            generatedImages = generatedImages.reshape(10, 32, 32,3)
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
            ax.imshow(img.reshape((32,32)), cmap='Greys_r')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    plt.show()