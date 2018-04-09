from keras.layers import Input, Dense, LeakyReLU, Reshape,Conv2DTranspose, Conv2D,Flatten,Dropout,\
    BatchNormalization, Activation, UpSampling2D, MaxPooling2D, ZeroPadding2D, Cropping2D
from keras.models import Model,load_model, Sequential
from keras.optimizers import SGD,RMSprop, Adam
from keras.utils import multi_gpu_model
import numpy as np
import pickle
from matplotlib import pyplot as plt
from keras.preprocessing import image
from lib.config.STLConfig import STLConfig
from lib.datareader.DataReaderForClassification import DataReader
from lib.utils.progressbar.ProgressBar import ProgressBar
from keras.initializers import RandomNormal

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)
config = STLConfig()

def builder_generator(isize=96, nz=100, nc=3, ngf=64):
    # cngf= ngf//2
    # tisize = isize
    # while tisize > 5:
    #     cngf = cngf * 2
    #     tisize = tisize // 2
    # _ = inputs = Input(shape=(nz,))
    # _ = Reshape((1,1,nz))(_)
    # _ = Conv2DTranspose(filters=cngf, kernel_size=tisize, strides=1, use_bias=False,
    #                     kernel_initializer = conv_init,
    #                     name = 'initial.{0}-{1}.convt'.format(nz, cngf))(_)
    # _ = BatchNormalization(axis=1, epsilon=1.01e-5, name = 'initial.{0}.batchnorm'.format(cngf))(_, training=1)
    # _ = Activation("relu", name = 'initial.{0}.relu'.format(cngf))(_)
    # csize, cndf = tisize, cngf
    #
    # while csize < isize//2:
    #     in_feat = cngf
    #     out_feat = cngf//2
    #     _ = Conv2DTranspose(filters=out_feat, kernel_size=4, strides=2, use_bias=False,
    #                         kernel_initializer = conv_init,
    #                         name = 'pyramid.{0}-{1}.convt'.format(in_feat, out_feat)
    #                         ) (_)
    #     _ = Cropping2D(cropping=1, name = 'pyramid.{0}.cropping'.format(in_feat) )(_)
    #     _ = BatchNormalization(axis=1, epsilon=1.01e-5, name = 'pyramid.{0}.batchnorm'.format(out_feat))(_, training=1)
    #     _ = Activation("relu", name = 'pyramid.{0}.relu'.format(out_feat))(_)
    #     csize, cngf = csize*2, cngf//2
    # _ = Conv2DTranspose(filters=nc, kernel_size=4, strides=2, use_bias=False,
    #                     kernel_initializer = conv_init,
    #                     name = 'final.{0}-{1}.convt'.format(cngf, nc)
    #                     )(_)
    # _ = Cropping2D(cropping=1, name = 'final.{0}.cropping'.format(nc) )(_)
    # outputs = Activation("tanh", name = 'final.{0}.tanh'.format(nc))(_)
    # return Model(inputs=inputs, outputs=outputs)

    noise = Input(shape=(nz,))

    network = Dense(units=1024)(noise)
    network = Activation('tanh')(network)
    network = Dense(units=128*8*8)(network)
    network = BatchNormalization()(network)
    network = Activation('tanh')(network)
    network = Reshape(target_shape=(8,8,128))(network)
    network = UpSampling2D()(network)

    network = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same')(network)
    network = BatchNormalization()(network)
    network = Activation('tanh')(network)


    network = Conv2DTranspose(filters=3, kernel_size=4, strides=3, padding='same')(network)
    network = Activation('tanh')(network)

    model = Model(inputs=noise, outputs=network)
    return model

def builder_discriminator(isize = 96, nc = 3, ndf = 64):
    # inputs = Input(shape=(isize, isize, nc))
    # network = ZeroPadding2D(name = 'initial.padding.{0}'.format(nc))(inputs)
    # network = Conv2D(filters=ndf, kernel_size=4, strides=2, use_bias=False,
    #                  kernel_initializer = conv_init,
    #                  name = 'initial.conv.{0}-{1}'.format(nc, ndf)
    #                  ) (network)
    # network = LeakyReLU(alpha=0.2, name = 'initial.relu.{0}'.format(ndf))(network)
    # csize, cndf = isize // 2, ndf
    # while csize > 5:
    #     in_feat = cndf
    #     out_feat = cndf*2
    #     network = ZeroPadding2D(name = 'pyramid.{0}.padding'.format(in_feat))(network)
    #     network = Conv2D(filters=out_feat, kernel_size=4, strides=2, use_bias=False,
    #                      kernel_initializer = conv_init,
    #                      name = 'pyramid.{0}-{1}.conv'.format(in_feat, out_feat)
    #                      ) (network)
    #     network = BatchNormalization(name = 'pyramid.{0}.batchnorm'.format(out_feat),
    #                                  axis=1, epsilon=1.01e-5)(network, training=1)
    #     network = LeakyReLU(alpha=0.2, name = 'pyramid.{0}.relu'.format(out_feat))(network)
    #     csize, cndf = csize//2, cndf*2
    # network = Conv2D(filters=1, kernel_size=csize, strides=1, use_bias=False,
    #                  name = 'final.{0}-{1}.conv'.format(cndf, 1)
    #                  ) (network)
    # outputs = Flatten()(network)
    # return Model(inputs=inputs, outputs=outputs)

    img     = Input(shape=(96,96,3))
    network = Conv2D(32, kernel_size=2, strides=2,  padding='same')(img)
    network = Activation('relu')(network)
    network = MaxPooling2D(pool_size=(2,2))(network)
    network = Conv2D(64, kernel_size=3, strides=2, padding="same")(network)
    # network = ZeroPadding2D(padding=((0,1),(0,1)))(network)
    network = Activation('relu')(network)
    network = BatchNormalization(momentum=0.8)(network)
    network = Conv2D(128, kernel_size=3, strides=2, padding="same")(network)
    network = Activation('relu')(network)
    network = BatchNormalization(momentum=0.8)(network)
    network = MaxPooling2D(pool_size=(2,2))(network)
    network = Conv2D(256, kernel_size=3, strides=1, padding="same")(network)
    network = Activation('relu')(network)
    network = Flatten()(network)
    network = Dense(1, activation='sigmoid')(network)
    model   = Model(inputs=img, outputs=network)

    return model

PHRASE = "TRAIN"
GPU_NUM = 1
batchSize = 50
epochs = 60
randomDim = 100
if PHRASE == "TRAIN":
    # adam = Adam(lr=0.0002, beta_1=0.5)
    reader = DataReader("../ganData/STL/")
    imageList, labelList = reader.readData(image_shape=(config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNEL),
                                           subFolder='1')

    discriminator = builder_discriminator()
    # discriminator = multi_gpu_model(discriminator, GPU_NUM)
    discriminator.compile(optimizer=SGD(lr=0.003),#RMSprop(lr=0.0003, clipvalue=1.0, decay=1e-8),
                          loss='binary_crossentropy')
    discriminator.trainable = False

    generator = builder_generator(nz=randomDim)
    generator.compile(optimizer=SGD(lr=0.003),#RMSprop(lr=0.0003, clipvalue=1.0, decay=1e-8),
                                    loss='binary_crossentropy')
    ganInput = Input(shape=(randomDim,))
    ganOutput = discriminator(generator(ganInput))
    dcgan = Model(inputs=ganInput, outputs=ganOutput)

    if GPU_NUM > 1:
        dcgan = multi_gpu_model(dcgan, GPU_NUM)
    dcgan.compile(loss='binary_crossentropy', optimizer=
    SGD(lr=0.003)#RMSprop(lr=0.0003, clipvalue=1.0, decay=1e-8)
                  )

    dLosses = []
    aLosses = []

    dloss = 0
    aloss = 0

    x_train = imageList
    y_train = labelList

    x_train = (x_train.reshape(
        (x_train.shape[0],) + (config.IMAGE_SIZE, config.IMAGE_SIZE, config.IMAGE_CHANNEL)).astype('float32')) / 255

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
