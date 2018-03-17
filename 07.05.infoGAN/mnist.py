from keras import layers as KLayers
from keras import models as KModels
from keras import optimizers as KOpts
from keras import backend as K
from keras_datareaders import mnistReader as reader
import numpy as np
from keras import utils as KUtils
from utils.progressbar.keras import ProgressBarCallback as bar
from matplotlib import pyplot as plt

PHRASE = "TRAIN"

if PHRASE == "TRAIN":
    def gaussian_loss(y_true, y_pred):

        mean = y_pred[0]
        log_stddev = y_pred[1]
        y_true = y_true[0]

        epsilon = (y_true - mean) / (K.exp(log_stddev) + K.epsilon())
        loss = (log_stddev + 0.5 * K.square(epsilon))

        return K.mean(loss)
    def build_generator():
        img = KLayers.Input(shape=(74,))
        network = KLayers.Dense(1024, activation="relu")(img)
        network = KLayers.BatchNormalization(momentum=0.8)(network)
        network = KLayers.Dense(128*7*7, activation="relu")(network)
        network = KLayers.BatchNormalization(momentum=0.8)(network)
        network = KLayers.Reshape((7,7,128))(network)
        network = KLayers.UpSampling2D()(network)
        network = KLayers.Conv2D(64, kernel_size=4, padding="same")(network)
        network = KLayers.Activation("relu")(network)
        network = KLayers.BatchNormalization(momentum=0.8)(network)
        network = KLayers.UpSampling2D()(network)
        network = KLayers.Conv2D(1, kernel_size=4, padding="same")(network)
        network = KLayers.Activation("tanh")(network)
        model = KModels.Model(inputs=img, outputs=network)
        model.compile(loss='binary_crossentropy', optimizer=adam)
        return model

    def build_discriminator():
        img = KLayers.Input(shape=(28,28,1))
        network = KLayers.Conv2D(64, kernel_size=2, strides=2, padding="same")(img)
        network = KLayers.LeakyReLU(alpha=0.2)(network)
        network = KLayers.Dropout(rate=0.25)(network)
        network = KLayers.Conv2D(128, kernel_size=4, strides=2, padding="same")(network)
        network = KLayers.LeakyReLU(alpha=0.2)(network)
        network = KLayers.Dropout(rate=0.25)(network)
        network = KLayers.Flatten()(network)
        network = KLayers.Dense(1024)(network)
        network = KLayers.LeakyReLU(alpha=0.2)(network)
        network = KLayers.BatchNormalization(momentum=0.8)(network)
        validity = KLayers.Dense(1, activation="sigmoid")(network)

        c_model = KLayers.Dense(128)(network)
        c_model = KLayers.LeakyReLU(alpha=0.2)(c_model)
        c_model = KLayers.BatchNormalization(momentum=0.8)(c_model)
        label = KLayers.Dense(10, activation="softmax")(c_model)

        def linmax(x):
            return K.maximum(x, -16)

        def linmax_shape(input_shape):
            return input_shape

        mean = KLayers.Dense(1, activation="linear")(c_model)
        log_stddev = KLayers.Dense(1)(c_model)
        log_stddev = KLayers.Lambda(linmax, output_shape=linmax_shape)(log_stddev)

        count = KLayers.concatenate([mean, log_stddev], axis=1)

        model = KModels.Model(inputs=img, outputs=[validity, label, count])
        model.compile(loss=['binary_crossentropy', 'categorical_crossentropy', gaussian_loss], optimizer=adam)
        return model

    def sample_generator_input(batch_size):
        sampled_noise = np.random.normal(0, 1, (batch_size, 62))
        sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
        sampled_labels = KUtils.to_categorical(sampled_labels, num_classes=10)
        sampled_cont = np.random.uniform(-1, 1, size=(batch_size, 2))
        return sampled_noise, sampled_labels, sampled_cont

    adam = KOpts.Adam(lr=0.0002,beta_1=0.5)
    epochs = 6000
    batch_size = 100
    (x_train, y_train), (_,_) = reader.read_mnist()

    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=3)
    y_train = y_train.reshape(-1,1)

    discriminator = build_discriminator()
    discriminator.trainable = False

    ganInput = KLayers.Input(shape=(74,))
    generator = build_generator()
    x = generator(ganInput)
    valid, target_label, target_count = discriminator(x)
    gan = KModels.Model(inputs=ganInput, outputs=[valid, target_label, target_count])
    gan.compile(loss=['binary_crossentropy', 'categorical_crossentropy', gaussian_loss], optimizer=adam)
    progBar = bar.ProgressBarGAN(1, epochs, "D Loss:%.3f,G Loss:%.3f")

    for epoch in range(epochs):
        sampled_noise, sampled_labels,sampled_count = sample_generator_input(batch_size)
        gen_input = np.concatenate([sampled_noise, sampled_labels, sampled_count], axis=1)
        generateImages = generator.predict(gen_input)

        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]

        fake = np.zeros((batch_size, 1))
        valid = np.ones((batch_size, 1))

        labels = KUtils.to_categorical(y_train[idx], num_classes=10)
        discriminator.trainable = True
        dloss = discriminator.train_on_batch(np.concatenate([imgs, generateImages]),
                                             [np.concatenate([valid, fake]),
                                              np.concatenate([labels, sampled_labels]),
                                              np.concatenate([sampled_count, sampled_count])])


        valid = np.ones((batch_size, 1))

        sampled_noise, sampled_labels, sampled_cont = sample_generator_input(batch_size)
        gen_input = np.concatenate([sampled_noise, sampled_labels, sampled_cont], axis=1)
        discriminator.trainable = False
        gloss = gan.train_on_batch(gen_input, [valid, sampled_labels, sampled_count])
        progBar.show(dloss[0], gloss[0])

        if epoch == 1 or epoch % 600 == 0:
            r, c = 10, 10

            fig, axs = plt.subplots(r, c)
            for i in range(r):
                sampled_noise, sampled_labels, sampled_cont = sample_generator_input(c)
                gen_input = np.concatenate([sampled_noise, sampled_labels, sampled_cont], axis=1)
                gen_imgs = generator.predict(gen_input)
                gen_imgs = 0.5 * gen_imgs + 0.5
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[j,:,:,0], cmap='gray')
                    axs[i,j].axis('off')
            fig.savefig("output/mnist_%d.png" % epoch)
            plt.close()
else:
    pass
