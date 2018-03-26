import os
import imageio
import numpy as np

from skimage.io import imread
from keras.layers import Input, Dense, LeakyReLU, Activation
from keras.models import Model
from keras.optimizers import RMSprop
from keras.utils import multi_gpu_model

from lib.models.sampler import generate_lut,sample_2d
from lib.models.visualize import GANDemoVisualizer
from lib.utils.progressbar.ProgressBar import ProgressBar

GPU_NUMS = 1
DIMENSION = 2
iterations = 3000
cuda = False
bs = 2000
z_dim = 2
input_path = "inputs/Z.jpg"

def build_generator():
    img =     Input(shape=(2,))
    network = Dense(50)(img)
    network = LeakyReLU(alpha=0.1)(network)
    network = Dense(2)(network)
    network = Activation("sigmoid")(network)
    model   = Model(inputs=img, outputs=network)
    return model

def build_discriminator():
    img =     Input(shape=(2,))
    network = Dense(50)(img)
    network = LeakyReLU(alpha=0.1)(network)
    network = Dense(1)(network)
    network = Activation("sigmoid")(network)
    model   = Model(inputs=img, outputs=network)
    if GPU_NUMS > 1:
        model = multi_gpu_model(model,GPU_NUMS)
    model.compile(optimizer=RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8), loss="binary_crossentropy")
    return model

density_img = imread(input_path, True)
lut_2d = generate_lut(density_img)
visualizer = GANDemoVisualizer('GAN 2D Example Visualization of {}'.format(input_path))

generator = build_generator()
discriminator = build_discriminator()
discriminator.trainable = False

ganInput = Input(shape=(z_dim,))
generator = build_generator()
x = generator(ganInput)
ganOutput = discriminator(x)
gan = Model(inputs=ganInput, outputs=ganOutput)
if GPU_NUMS > 1:
    gan = multi_gpu_model(gan,GPU_NUMS)
gan.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8))
progBar = ProgressBar(1, iterations, "D Loss:%.3f,G Loss:%.3f")

for epoch_iter in range(1, iterations+1):
    for index in range(20):
        real_samples = sample_2d(lut_2d, bs)
        # print(real_samples.shape)

        noise = np.random.normal(-1, 1, size=[bs, z_dim])
        generateImage = generator.predict(noise)

        discriminator.trainable = True
        yDis = np.zeros(2*bs)
        yDis[:bs] = 1
        d_loss = discriminator.train_on_batch(
            np.concatenate((real_samples, generateImage)), yDis)
    for index in range(1):
        noise = np.random.normal(-1, 1, size=[bs, z_dim])
        generateImage = generator.predict(noise)
        yGen = np.ones(bs)
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, yGen)

    progBar.show(d_loss, g_loss)

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

filenames=sorted((os.path.join(output_dir, fn) for fn in os.listdir(output_dir) if fn.endswith('.png')))
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('{}.gif'.format(filename[:filename.rfind('.')]), images,duration=0.1)