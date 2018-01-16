import os
from skimage import io
import torch
import torch.nn as nn
from torch.autograd import Variable
from keras_commons import sampler as sampler
from keras_commons import visualize as visualizer
import torch.nn.functional as F
from torch import optim as tOpt
PHRASE = "TRAIN"

DIMENSION = 2

cuda = False
bs = 2000
z_dim = 2
input_path = "inputs/Z.jpg"

density_img = io.imread(input_path, True)
lut_2d = sampler.generate_lut(density_img)

visualizer = visualizer.GANDemoVisualizer('GAN 2D Example Visualization of {}'.format(input_path))

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
    generator = SimpleMLP(input_size=z_dim, hidden_size=50, output_size=DIMENSION)
    discriminator = SimpleMLP(input_size=DIMENSION, hidden_size=100, output_size=1)
    if cuda:
        generator.cuda()
    discriminator.cuda()
    criterion = nn.BCELoss()

    d_optimizer = tOpt.Adadelta(discriminator.parameters(), lr=1)
    g_optimizer = tOpt.Adadelta(generator.parameters(), lr=1)

    for train_iter in range(2000):
        for d_index in range(3):
            # 1. Train D on real+fake
            discriminator.zero_grad()

            #  1A: Train D on real
            real_samples = sampler.sample_2d(lut_2d, bs)
            d_real_data = Variable(torch.Tensor(real_samples))
            if cuda:
                d_real_data = d_real_data.cuda()
            d_real_decision = discriminator(d_real_data)
            labels = Variable(torch.ones(bs))
            if cuda:
                labels = labels.cuda()
            d_real_loss = criterion(d_real_decision, labels)  # ones = true

            #  1B: Train D on fake
            latent_samples = torch.randn(bs, z_dim)
            d_gen_input = Variable(latent_samples)
            if cuda:
                d_gen_input = d_gen_input.cuda()
            d_fake_data = generator(d_gen_input).detach()  # detach to avoid training G on these labels
            d_fake_decision = discriminator(d_fake_data)
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

            latent_samples = torch.randn(bs, z_dim)
            g_gen_input = Variable(latent_samples)
            if cuda:
                g_gen_input = g_gen_input.cuda()
            g_fake_data = generator(g_gen_input)
            g_fake_decision = discriminator(g_fake_data)
            labels = Variable(torch.ones(bs))
            if cuda:
                labels = labels.cuda()
            g_loss = criterion(g_fake_decision, labels)  # we want to fool, so pretend it's all genuine

            g_loss.backward()
            g_optimizer.step()  # Only optimizes G's parameters

        if train_iter % 100 == 0:
            loss_d_real = d_real_loss.data.cpu().numpy()[0] if cuda else d_real_loss.data.numpy()[0]
            loss_d_fake = d_fake_loss.data.cpu().numpy()[0] if cuda else d_fake_loss.data.numpy()[0]
            loss_g = g_loss.data.cpu().numpy()[0] if cuda else g_loss.data.numpy()[0]

            msg = 'Iteration {}: D_loss(real/fake): {:.6g}/{:.6g} G_loss: {:.6g}'.format(train_iter, loss_d_real, loss_d_fake, loss_g)
            print(msg)

            gen_samples = g_fake_data.data.cpu().numpy() if cuda else g_fake_data.data.numpy()

            visualizer.draw(real_samples, gen_samples, msg, show=False)

            if True:
                filename = input_path.split(os.sep)[-1]
                output_dir = 'gan_training_{}'.format(filename[:filename.rfind('.')])
                os.system('mkdir -p {}'.format(output_dir))
                export_filepath = os.sep.join([output_dir, 'iter_{:0>6d}.png'.format(train_iter)])
                visualizer.savefig(export_filepath)
    torch.save(generator, "generator.pkl")
    torch.save(discriminator, "discriminator.pkl")
    if not True:
        visualizer.show()
else:
    generator = torch.load("generator.pkl")
    real_samples = sampler.sample_2d(lut_2d, bs)
    latent_samples = torch.randn(bs, z_dim)
    g_gen_input = Variable(latent_samples)
    g_fake_data = generator(g_gen_input)
    gen_samples = g_fake_data.data.numpy()
    visualizer.draw(real_samples, gen_samples, "", show=False)
    filename = input_path.split(os.sep)[-1]
    output_dir = 'gan_training_{}'.format(filename[:filename.rfind('.')])
    os.system('mkdir -p {}'.format(output_dir))
    export_filepath = os.sep.join([output_dir, 'iter_{:0>6d}.png'.format(0)])
    visualizer.savefig(export_filepath)

