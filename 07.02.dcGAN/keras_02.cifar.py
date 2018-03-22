from keras.layers import Input, Dense, LeakyReLU, Reshape,Conv2DTranspose, Conv2D,Flatten,Dropout
from keras.models import Model,load_model
from keras.optimizers import Adam,RMSprop
from keras.utils import multi_gpu_model
import numpy as np
import pickle
from matplotlib import pyplot as plt
from keras.preprocessing import image
from lib.config.cifarConfig import Cifar10Config
from lib.datareader.DataReaderForClassification import DataReader
from lib.utils.progressbar.ProgressBar import ProgressBar

config = Cifar10Config()

def builder_generator(randomDim):
    generator_input = Input(shape=(randomDim,))
    network = Dense(128 * 16 * 16)(generator_input)
    network = LeakyReLU()(network)
    network = Reshape((16, 16, 128))(network)

    network = Conv2D(256, 5, padding='same')(network)
    network = LeakyReLU()(network)

    network = Conv2DTranspose(256, 4, strides=2, padding='same')(network)
    network = LeakyReLU()(network)

    # Few more conv layers
    network = Conv2D(256, 5, padding='same')(network)
    network = LeakyReLU()(network)
    network = Conv2D(256, 5, padding='same')(network)
    network = LeakyReLU()(network)

    network = Conv2D(config.IMAGE_CHANNEL, 7, activation='tanh', padding='same')(network)
    generator = Model(generator_input, network)
    # generator.summary()
    return generator

def builder_discriminator():
    discriminator_input = Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNEL))
    network = Conv2D(128, 3)(discriminator_input)
    network = LeakyReLU()(network)
    network = Conv2D(128, 4, strides=2)(network)
    network = LeakyReLU()(network)
    network = Conv2D(128, 4, strides=2)(network)
    network = LeakyReLU()(network)
    network = Conv2D(128, 4, strides=2)(network)
    network = LeakyReLU()(network)
    network = Flatten()(network)
    network = Dropout(0.4)(network)
    network = Dense(1, activation='sigmoid')(network)

    discriminator = Model(discriminator_input, network)
    # discriminator.summary()
    return discriminator



PHRASE = "TRAIN"
GPU_NUM = 1
batchSize = 100
epochs = 60
if PHRASE == "TRAIN":
    # adam = Adam(lr=0.0002, beta_1=0.5)
    reader = DataReader()
    imageList, labelList = reader.readData()

    discriminator = builder_discriminator()
    # discriminator = multi_gpu_model(discriminator, GPU_NUM)
    discriminator.compile(optimizer=RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8), loss='binary_crossentropy')
    discriminator.trainable = False

    generator = builder_generator(100)
    # generator.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    ganInput = Input(shape=(100,))
    ganOutput = discriminator(generator(ganInput))
    dcgan = Model(inputs=ganInput, outputs=ganOutput)

    if GPU_NUM > 1:
        dcgan = multi_gpu_model(dcgan, GPU_NUM)
    dcgan.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8))

    dLosses = []
    aLosses = []

    dloss = 0
    aloss = 0

    x_train = imageList
    y_train = labelList
    x_train = imageList[y_train.flatten() == 7]

    x_train = x_train.reshape(
        (x_train.shape[0],) + (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNEL)).astype('float32') / 255.

    batchCount = x_train.shape[0] // batchSize
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)
    progBar = ProgressBar(epochs, batchCount, "D Loss:%.3f;G Loss:%.3f")
    samples_image = []
    start = 0

    for epoch in range(1, (epochs + 1)):
        for _ in range(batchCount):
            noise = np.random.normal(size=(batchSize, 100))
            generatedImages = generator.predict(noise)

            imageBatch = x_train[np.random.randint(0, x_train.shape[0], size=batchSize)]

            combined_images = np.concatenate([generatedImages, imageBatch])

            labels = np.concatenate([np.ones((batchSize, 1)),
                                     np.zeros((batchSize, 1))])
            labels += 0.05 * np.random.random(labels.shape)

            d_loss = discriminator.train_on_batch(combined_images, labels)

            noise = np.random.normal(size=[batchSize, 100])
            yGen = np.zeros(batchSize)
            aloss = dcgan.train_on_batch(noise, yGen)
            progBar.show(d_loss, aloss)

        dLosses.append(dloss)
        aLosses.append(aloss)
        if epoch == 1 or epoch % 5 == 0:
            samples_image.append(generatedImages)
            img = image.array_to_img(generatedImages[0] * 255., scale=False)
            img.save('generated_horse' + str(epoch) + '.png')

    with open('train_samples.pkl', 'wb') as f:
        pickle.dump(samples_image, f)

    generator.save('cifar_generator.h5')
else:
    generator = load_model("cifar_generator.h5")
    noise = np.random.normal(size=[batchSize, 100])
    generated_images = generator.predict(noise)

    plt.figure(num='astronaut',figsize=(8,8))
    # generated_images = generated_images * 2 - 1
    for index, img in enumerate(generated_images):
        img = image.array_to_img(img * 255., scale=False)
        plt.subplot(10,10,index+1)
        plt.imshow(img)

    plt.show()
