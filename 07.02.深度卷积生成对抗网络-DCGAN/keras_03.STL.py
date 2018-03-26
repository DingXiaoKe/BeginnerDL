from keras.layers import Input, Dense, LeakyReLU, Reshape,Conv2DTranspose, Conv2D,Flatten,Dropout,\
    BatchNormalization, Activation
from keras.models import Model,load_model
from keras.optimizers import SGD,RMSprop, Adam
from keras.utils import multi_gpu_model
import numpy as np
import pickle
from matplotlib import pyplot as plt
from keras.preprocessing import image
from lib.config.STLConfig import STLConfig
from lib.datareader.DataReaderForClassification import DataReader
from lib.utils.progressbar.ProgressBar import ProgressBar

config = STLConfig()

def builder_generator(randomDim):
    generator_input = Input(shape=(randomDim,))

    network = Reshape((1,1,randomDim))(generator_input)

    # 1,1,100 => 4,4,512
    network = Conv2DTranspose(filters=64*8, kernel_size=4, strides=4,padding='same')(network)
    network = BatchNormalization()(network)
    network = Activation('relu')(network)

    # 4,4,512 => 8,8,256
    network = Conv2DTranspose(filters=64*4, kernel_size=4, strides=2,padding='same')(network)
    network = BatchNormalization()(network)
    network = Activation('relu')(network)

    # 8,8,256 => 16,16,128
    network = Conv2DTranspose(filters=64*2, kernel_size=4, strides=2,padding='same')(network)
    network = BatchNormalization()(network)
    network = Activation('relu')(network)

    # 16,16,128 => 32,32,64
    network = Conv2DTranspose(filters=64*1, kernel_size=4, strides=2,padding='same')(network)
    network = BatchNormalization()(network)
    network = Activation('relu')(network)

    # 32,32,64 => 96,96,3
    network = Conv2DTranspose(filters=3, kernel_size=4, strides=3,padding='same')(network)
    network = Activation('tanh')(network)

    # network = Dense(64 * 24 * 24)(generator_input)
    # network = LeakyReLU()(network)
    # network = Reshape((24, 24, 64))(network)
    #
    # network = Conv2DTranspose(128, 4, strides=2, padding='same')(network)
    # network = LeakyReLU()(network)
    #
    # network = Conv2D(256, 5, padding='same')(network)
    # network = LeakyReLU()(network)
    # network = Conv2DTranspose(256, 4, strides=2, padding='same')(network)
    # network = LeakyReLU()(network)
    # network = Conv2D(256, 5, padding='same')(network)
    # network = LeakyReLU()(network)
    # network = Conv2D(256, 5, padding='same')(network)
    # network = LeakyReLU()(network)
    # network = Conv2D(config.IMAGE_CHANNEL, 7, activation='tanh', padding='same')(network)

    generator = Model(generator_input, network)
    return generator

def builder_discriminator():
    discriminator_input = Input(shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNEL))

    # 96,96,3 => 32,32,64
    network =Conv2D(filters=64, kernel_size=4, strides=3, padding='same')(discriminator_input)
    network = LeakyReLU(alpha=0.2)(network)
    network = BatchNormalization()(network)

    # 32,32,64 => 16,16,128
    network = Conv2D(filters=64*2, kernel_size=4, strides=2, padding='same')(network)
    network = LeakyReLU(alpha=0.2)(network)
    network = BatchNormalization()(network)

    # 16,16,128 => 8,8,256
    network = Conv2D(filters=64*4, kernel_size=4, strides=2, padding='same')(network)
    network = LeakyReLU(alpha=0.2)(network)
    network = BatchNormalization()(network)

    # 8,8,256 => 4,4,512
    network = Conv2D(filters=64*8, kernel_size=4, strides=2, padding='same')(network)
    network = LeakyReLU(alpha=0.2)(network)
    network = BatchNormalization()(network)

    # 4,4,512 =>
    network = Conv2D(filters=1, kernel_size=4, strides=4, padding='same')(network)
    network = Flatten()(network)
    network = Dense(1)(network)

    # network = Conv2D(128, 3)(discriminator_input)
    # network = LeakyReLU()(network)
    # network = Conv2D(128, 4, strides=2)(network)
    # network = LeakyReLU()(network)
    # network = Conv2D(128, 4, strides=2)(network)
    # network = LeakyReLU()(network)
    # network = Conv2D(128, 4, strides=2)(network)
    # network = LeakyReLU()(network)
    # network = Flatten()(network)
    # network = Dropout(0.4)(network)
    # network = Dense(1)(network)

    discriminator = Model(discriminator_input, network)

    return discriminator

PHRASE = "TRAIN"
GPU_NUM = 2
batchSize = 50
epochs = 60
randomDim = 100
if PHRASE == "TRAIN":
    # adam = Adam(lr=0.0002, beta_1=0.5)
    reader = DataReader("../data/STL/")
    imageList, labelList = reader.readData(image_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNEL),
                                           subFolder='1')

    discriminator = builder_discriminator()
    # discriminator = multi_gpu_model(discriminator, GPU_NUM)
    discriminator.compile(optimizer=RMSprop(lr=0.00005, clipvalue=1.0, decay=1e-8), loss='binary_crossentropy')
    discriminator.trainable = False

    generator = builder_generator(randomDim)
    # generator.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    ganInput = Input(shape=(randomDim,))
    ganOutput = discriminator(generator(ganInput))
    dcgan = Model(inputs=ganInput, outputs=ganOutput)

    if GPU_NUM > 1:
        dcgan = multi_gpu_model(dcgan, GPU_NUM)
    dcgan.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.00005, clipvalue=1.0, decay=1e-8))

    dLosses = []
    aLosses = []

    dloss = 0
    aloss = 0

    x_train = imageList
    y_train = labelList

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
            img.save('generated_airplane' + str(epoch) + '.png')

    with open('train_samples.pkl', 'wb') as f:
        pickle.dump(samples_image, f)

    generator.save('stl_generator.h5')
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
