from skimage import io
from keras_commons import sampler as sampler
from keras_commons import visualize as visualizer
from keras import layers as KLayers
from keras import models as KModels
from keras import optimizers as KOpts
import numpy as np
from keras import losses as KLosses
from keras_callbacks import ProgressBarCallback as bar
import os
import torch
import math
from keras import backend as K

# Iteration 0: D_loss(real/fake): 0.666903/0.714351 G_loss: 0.673248
# Iteration 100: D_loss(real/fake): 0.644591/0.742094 G_loss: 0.73878
# Iteration 200: D_loss(real/fake): 0.648646/0.735631 G_loss: 0.742899
# Iteration 300: D_loss(real/fake): 0.650042/0.733398 G_loss: 0.741042
# Iteration 400: D_loss(real/fake): 0.667946/0.715312 G_loss: 0.734752
# Iteration 500: D_loss(real/fake): 0.653243/0.73298 G_loss: 0.738096
# Iteration 600: D_loss(real/fake): 0.679417/0.70518 G_loss: 0.701225
# Iteration 700: D_loss(real/fake): 0.685357/0.698812 G_loss: 0.693155
# Iteration 800: D_loss(real/fake): 0.665757/0.71943 G_loss: 0.726154
# Iteration 900: D_loss(real/fake): 0.677652/0.706955 G_loss: 0.705668
# Iteration 1000: D_loss(real/fake): 0.666454/0.718688 G_loss: 0.721987
# Iteration 1100: D_loss(real/fake): 0.677197/0.706994 G_loss: 0.717063
# Iteration 1200: D_loss(real/fake): 0.686249/0.698309 G_loss: 0.697095
# Iteration 1300: D_loss(real/fake): 0.716442/0.667566 G_loss: 0.674883
# Iteration 1400: D_loss(real/fake): 0.669951/0.715769 G_loss: 0.714323
# Iteration 1500: D_loss(real/fake): 0.70815/0.678253 G_loss: 0.672678
# Iteration 1600: D_loss(real/fake): 0.694573/0.690822 G_loss: 0.699048
# Iteration 1700: D_loss(real/fake): 0.687652/0.698034 G_loss: 0.698886
# Iteration 1800: D_loss(real/fake): 0.704925/0.68092 G_loss: 0.685825
# Iteration 1900: D_loss(real/fake): 0.677409/0.70847 G_loss: 0.706711

opt = KOpts.Adadelta(lr = 1)

def build_generator():
    img = KLayers.Input(shape=(2,))
    network = KLayers.Dense(50)(img)
    network = KLayers.LeakyReLU(alpha=0.1)(network)
    network = KLayers.Dense(2)(network)
    network = KLayers.Activation("sigmoid")(network)

    model = KModels.Model(inputs=img, outputs=network)
    model.compile(optimizer=opt, loss="binary_crossentropy")

    return model

def build_discriminator():
    img = KLayers.Input(shape=(2,))
    network = KLayers.Dense(50)(img)
    network = KLayers.LeakyReLU(alpha=0.1)(network)
    network = KLayers.Dense(1)(network)
    network = KLayers.Activation("sigmoid")(network)

    model = KModels.Model(inputs=img, outputs=network)
    model.compile(optimizer=opt, loss="binary_crossentropy")

    return model

DIMENSION = 2

cuda = False
bs = 2000
z_dim = 2
input_path = "inputs/Z.jpg"

density_img = io.imread(input_path, True)
lut_2d = sampler.generate_lut(density_img)
visualizer = visualizer.GANDemoVisualizer('GAN 2D Example Visualization of {}'.format(input_path))

generator = build_generator()
discriminator = build_discriminator()
discriminator.trainable = False

ganInput = KLayers.Input(shape=(z_dim,))
generator = build_generator()
x = generator(ganInput)
ganOutput = discriminator(x)
gan = KModels.Model(inputs=ganInput, outputs=ganOutput)
gan.compile(loss='binary_crossentropy', optimizer=opt)
progBar = bar.ProgressBarGAN(1, 2000, "D Loss:%.3f,G Loss:%.3f")

for epoch_iter in range(1, 2001):

    real_samples = sampler.sample_2d(lut_2d, bs)
    # print(real_samples.shape)

    noise = torch.randn(bs, z_dim) # np.random.normal(-1, 1, size=[bs, z_dim])
    generateImage = generator.predict(noise.numpy())

    discriminator.trainable = True
    yDis = np.zeros(2*bs)
    yDis[:bs] = 1
    d_loss = discriminator.train_on_batch(
        np.concatenate((real_samples, generateImage)), yDis)

    noise = torch.randn(bs, z_dim)#np.random.normal(-1, 1, size=[bs, z_dim])
    generateImage = generator.predict(noise.numpy())
    yGen = np.ones(bs)
    discriminator.trainable = False
    g_loss = gan.train_on_batch(noise, yGen)

    progBar.show(d_loss, g_loss)

    if epoch_iter % 300 == 0:
        K.set_value(opt.lr, 0.1 * K.get_value(opt.lr))

    if epoch_iter % 100 == 0:

        loss_g = g_loss

        msg = ""

        visualizer.draw(real_samples, generateImage, msg, show=False)

        if True:
            filename = input_path.split(os.sep)[-1]
            output_dir = 'gan_training_{}'.format(filename[:filename.rfind('.')])
            os.system('mkdir -p {}'.format(output_dir))
            export_filepath = os.sep.join([output_dir, 'iter_{:0>6d}.png'.format(epoch_iter)])
            visualizer.savefig(export_filepath)