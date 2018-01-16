from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from keras_config import mnist as config
from keras_datareaders import mnistReader as reader
import pickle
from keras_callbacks import ProgressBarCallback as bar
cfg = config.MNISTConfig()

def build_discriminator():
    model = Sequential()
    model.add(Dense(512, input_dim=cfg.ENCODED_DIM))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1, activation="sigmoid"))
    model.summary()
    encoded_repr = Input(shape=(cfg.ENCODED_DIM, ))
    validity = model(encoded_repr)
    return Model(encoded_repr, validity)

def build_generator(img_shape):
    # Encoder
    encoder = Sequential()

    encoder.add(Flatten(input_shape=img_shape))
    encoder.add(Dense(512))
    encoder.add(LeakyReLU(alpha=0.2))
    encoder.add(BatchNormalization(momentum=0.8))
    encoder.add(Dense(512))
    encoder.add(LeakyReLU(alpha=0.2))
    encoder.add(BatchNormalization(momentum=0.8))
    encoder.add(Dense(cfg.ENCODED_DIM))

    encoder.summary()

    img = Input(shape=img_shape)
    encoded_repr = encoder(img)

    # Decoder
    decoder = Sequential()

    decoder.add(Dense(512, input_dim=cfg.ENCODED_DIM))
    decoder.add(LeakyReLU(alpha=0.2))
    decoder.add(BatchNormalization(momentum=0.8))
    decoder.add(Dense(512))
    decoder.add(LeakyReLU(alpha=0.2))
    decoder.add(BatchNormalization(momentum=0.8))
    decoder.add(Dense(np.prod(img_shape), activation='tanh'))
    decoder.add(Reshape(img_shape))

    decoder.summary()

    gen_img = decoder(encoded_repr)

    return Model(img, [encoded_repr, gen_img])

def view_samples(epoch, samples):
    """
    epoch代表第几次迭代的图像
    samples为我们的采样结果
    """
    fig, axes = plt.subplots(figsize=(7,7), nrows=5, ncols=5, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]): # 这里samples[epoch][1]代表生成的图像结果，而[0]代表对应的logits
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')

    return fig, axes

img_shape = (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, cfg.IMAGE_CHANNEL)
img = Input(shape=img_shape)

optimizer = Adam(cfg.LEARNINGRATE_LR, cfg.LEARNINGRATE_BETA_1)

generator = build_generator(img_shape)
generator.compile(loss=['mse', 'binary_crossentropy'],
                       optimizer=optimizer)

discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

encoded_repr, reconstructed_img = generator(img)
discriminator.trainable = False
validity = discriminator(encoded_repr)

adversarial_autoencoder = Model(img, [reconstructed_img, validity])
adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
                                     loss_weights=[0.999, 0.001],
                                     optimizer=optimizer)

(x_train, y_train), (x_test, y_test) = reader.read_mnist('../data/mnist.npz')
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=3)

half_batch = int(cfg.BATCH_SIZE / 2)

samples_image = []
progressBar = bar.ProgressBarGAN(1, cfg.EPOCH_NUM, "D loss: %.3f, acc: %.2f%% - G loss: %.3f, mse: %.2f")
for epoch in range(cfg.EPOCH_NUM):
    # Select a random half batch of images
    idx = np.random.randint(0, x_train.shape[0], half_batch)
    imgs = x_train[idx]

    # Generate a half batch of new images
    latent_fake, gen_imgs = generator.predict(imgs)

    latent_real = np.random.normal(size=(half_batch, cfg.ENCODED_DIM))

    valid = np.ones((half_batch, 1))
    fake = np.zeros((half_batch, 1))

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(latent_real, valid)
    d_loss_fake = discriminator.train_on_batch(latent_fake, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Select a random half batch of images
    idx = np.random.randint(0, x_train.shape[0], cfg.BATCH_SIZE)
    imgs = x_train[idx]

    # Generator wants the discriminator to label the generated representations as valid
    valid_y = np.ones((cfg.BATCH_SIZE, 1))

    # Train the generator
    g_loss = adversarial_autoencoder.train_on_batch(imgs, [imgs, valid_y])

    # Plot the progress
    progressBar.show(d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1])
    # print ("%d [D loss: %.3f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

    # If at save interval => save generated image samples
    idx = np.random.randint(0, x_train.shape[0], 25)
    imgs = x_train[idx]
    samples_image.append(imgs)

with open('train_samples.pkl', 'wb') as f:
    pickle.dump(samples_image, f)

with open('train_samples.pkl', 'rb') as f:
    samples = pickle.load(f)
    # view_samples(-1, samples)
    epoch_idx = [0, 5, 10, 20, 40, 60, 80, 100, 150, 250] # 一共300轮，不要越界
    show_imgs = []
    for i in epoch_idx:
        show_imgs.append(samples[i])

    # 指定图片形状
    rows, cols = 10, 25
    fig, axes = plt.subplots(figsize=(30,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

    idx = range(0, cfg.EPOCH_NUM, int(cfg.EPOCH_NUM/rows))

    for sample, ax_row in zip(show_imgs, axes):
        for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
            ax.imshow(img.reshape((28,28)), cmap='Greys_r')
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

    plt.show()







