#!/usr/bin/env python
# Conditional Generative Adversarial Networks (GAN) example with 2D samples in PyTorch.
import os
import numpy
from skimage import io
import torch
import torch.nn as nn
from torch.autograd import Variable
from keras_commons import sampler as sampler
from keras_commons import visualize as visualizer
import torch.nn.functional as F
from torch import optim as tOpts
from utils.progressbar.keras import ProgressBarCallback as bar
import imageio
PHRASE = "TRAIN"
DIMENSION = 2
cuda = False
bs = 2000
iterations = 3000
z_dim = 2
input_path = "inputs/binary"
image_paths = [os.sep.join([input_path, x]) for x in os.listdir(input_path)]
density_imgs = [io.imread(x, True) for x in image_paths]
luts_2d = [sampler.generate_lut(x) for x in density_imgs]
# Sampling based on visual density, a too small batch size may result in failure with conditions
pix_sums = [numpy.sum(x) for x in density_imgs]
total_pix_sums = numpy.sum(pix_sums)
c_indices = [0] + [int(sum(pix_sums[:i+1])/total_pix_sums*bs+0.5) for i in range(len(pix_sums)-1)] + [bs]
c_dim = len(luts_2d)
visualizer = visualizer.CGANDemoVisualizer('Conditional GAN 2D Example Visualization of {}'.format(input_path))

if PHRASE == "TRAIN":
    class SimpleMLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(SimpleMLP, self).__init__()
            self.map1 = nn.Linear(input_size, hidden_size)
            self.map2 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = F.leaky_relu(self.map1(x), 0.1)
            return F.sigmoid(self.map2(x))

    class DeepMLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(DeepMLP, self).__init__()
            self.map1 = nn.Linear(input_size, hidden_size)
            self.map2 = nn.Linear(hidden_size, hidden_size)
            self.map3 = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            x = F.leaky_relu(self.map1(x), 0.1)
            x = F.leaky_relu(self.map2(x), 0.1)
            return F.sigmoid(self.map3(x))

    generator = SimpleMLP(input_size=z_dim+c_dim, hidden_size=50, output_size=DIMENSION)
    discriminator = SimpleMLP(input_size=DIMENSION+c_dim, hidden_size=100, output_size=1)

    if cuda:
        generator.cuda()
        discriminator.cuda()
    criterion = nn.BCELoss()

    d_optimizer = tOpts.Adadelta(discriminator.parameters(), lr=1)
    g_optimizer = tOpts.Adadelta(generator.parameters(), lr=1)

    y = numpy.zeros((bs, c_dim))
    for i in range(c_dim):
        y[c_indices[i]:c_indices[i + 1], i] = 1  # conditional labels, one-hot encoding
    y = Variable(torch.Tensor(y))
    if cuda:
        y = y.cuda()
    progBar = bar.ProgressBarGAN(1, iterations, "D Loss:(real/fake) %.3f/%.3f,G Loss:%.3f")
    for train_iter in range(1, iterations + 1):
        for d_index in range(3):
            # 1. Train D on real+fake
            discriminator.zero_grad()

            #  1A: Train D on real samples with conditions
            real_samples = numpy.zeros((bs, DIMENSION))
            for i in range(c_dim):
                real_samples[c_indices[i]:c_indices[i+1], :] = sampler.sample_2d(luts_2d[i], c_indices[i+1]-c_indices[i])

            # first c dimensions is the condition inputs, the last 2 dimensions are samples
            real_samples = Variable(torch.Tensor(real_samples))
            if cuda:
                real_samples = real_samples.cuda()
            d_real_data = torch.cat([y, real_samples], 1)
            if cuda:
                d_real_data = d_real_data.cuda()
            d_real_decision = discriminator(d_real_data)
            labels = Variable(torch.ones(bs))
            if cuda:
                labels = labels.cuda()
            d_real_loss = criterion(d_real_decision, labels)  # ones = true

            #  1B: Train D on fake
            latent_samples = Variable(torch.randn(bs, z_dim))
            if cuda:
                latent_samples = latent_samples.cuda()
            # first c dimensions is the condition inputs, the last z_dim dimensions are latent samples
            d_gen_input = torch.cat([y, latent_samples], 1)
            d_fake_data = generator(d_gen_input).detach()  # detach to avoid training G on these labels
            conditional_d_fake_data = torch.cat([y, d_fake_data], 1)
            if cuda:
                conditional_d_fake_data = conditional_d_fake_data.cuda()
            d_fake_decision = discriminator(conditional_d_fake_data)
            labels = Variable(torch.zeros(bs))
            if cuda:
                labels = labels.cuda()
            d_fake_loss = criterion(d_fake_decision, labels)  # zeros = fake

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()

            d_optimizer.step()     # Only optimizes D's parameters; changes based on stored gradients from backward()

        for g_index in range(1):
            # 2. Train G on D's response (but DO NOT train D on these labels)
            generator.zero_grad()

            latent_samples = Variable(torch.randn(bs, z_dim))
            if cuda:
                latent_samples = latent_samples.cuda()
            g_gen_input = torch.cat([y, latent_samples], 1)
            g_fake_data = generator(g_gen_input)
            conditional_g_fake_data = torch.cat([y, g_fake_data], 1)
            g_fake_decision = discriminator(conditional_g_fake_data)
            labels = Variable(torch.ones(bs))
            if cuda:
                labels = labels.cuda()
            g_loss = criterion(g_fake_decision, labels)  # we want to fool, so pretend it's all genuine

            g_loss.backward()
            g_optimizer.step()  # Only optimizes G's parameters

        loss_d_real = d_real_loss.data.cpu().numpy()[0] if cuda else d_real_loss.data.numpy()[0]
        loss_d_fake = d_fake_loss.data.cpu().numpy()[0] if cuda else d_fake_loss.data.numpy()[0]
        loss_g = g_loss.data.cpu().numpy()[0] if cuda else g_loss.data.numpy()[0]
        progBar.show(loss_d_real, loss_d_fake, loss_g)
        if train_iter == 1 or train_iter % 100 == 0:
            msg = 'Iteration {}: D_loss(real/fake): {:.6g}/{:.6g} G_loss: {:.6g}'.format(train_iter, loss_d_real, loss_d_fake, loss_g)

            real_samples_with_y = d_real_data.data.cpu().numpy() if cuda else d_real_data.data.numpy()
            gen_samples_with_y = conditional_g_fake_data.data.cpu().numpy() if cuda else conditional_g_fake_data.data.numpy()

            visualizer.draw(real_samples_with_y, gen_samples_with_y, msg, show=False)

            if True:
                filename = input_path.split(os.sep)[-1]
                output_dir = 'cgan_training_{}'.format(filename)
                os.system('mkdir -p {}'.format(output_dir))
                export_filepath = os.sep.join([output_dir, 'iter_{:0>6d}.png'.format(train_iter)])
                visualizer.savefig(export_filepath)
    torch.save(generator, "generator.pkl")
    torch.save(discriminator, "discriminator.pkl")

    filenames=sorted((os.path.join(output_dir, fn) for fn in os.listdir(output_dir) if fn.endswith('.png')))
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('{}.gif'.format(filename[:filename.rfind('.')]), images,duration=0.1)
    if not True:
        visualizer.show()
