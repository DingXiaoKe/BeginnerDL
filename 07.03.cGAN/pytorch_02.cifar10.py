# coding=utf-8
import torch
import torchvision
import numpy as np
import os

from torch.nn import BCELoss, Module, Conv2d, ConvTranspose2d, Linear, BatchNorm2d,DataParallel
from torch.nn.functional import relu, leaky_relu, tanh, sigmoid
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from torchvision.transforms import Resize, Compose,ToTensor
from torch.optim import Adam
from torch.autograd import Variable

from lib.config.cifarConfig import Cifar10Config
from lib.datareader.pytorch.cifar import Cifar10DataSet
from lib.utils.progressbar.ProgressBar import ProgressBar

GPU_NUMS = 1
BATCH_SIZE = 100
EPOCHS = 20
NOISE_DIM = 100
LR = 1e-3
config = Cifar10Config()

'''
生成器
input => [BATCH_SIZE, NOISE_DIM]
label => [BATCH_SIZE, 10]
'''
class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gc_full = Linear(NOISE_DIM + 10, 512)

        self.dconv1 = ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.dconv2 = ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.dconv3 = ConvTranspose2d(128, 64, kernel_size=3, stride=2)
        self.dconv4 = ConvTranspose2d(64, 3, kernel_size=3, stride=2, output_padding=1)

        self.bn4 = BatchNorm2d(64)
        self.bn3 = BatchNorm2d(128)
        self.bn2 = BatchNorm2d(256)
        self.bn1 = BatchNorm2d(512)

    def forward(self, input, label):
        network = torch.cat([input, label], dim=1)

        network = self.gc_full(network)
        network = relu(network)
        network = network.view(-1, 512, 1, 1)
        network = relu(self.bn2(self.dconv1(network)))
        network = relu(self.bn3(self.dconv2(network)))
        network = relu(self.bn4(self.dconv3(network)))
        network = sigmoid(self.dconv4(network))

        return network
'''
判别器
'''
class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=2)
        self.conv2 = Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv3 = Conv2d(128, 256, kernel_size=3, stride=2)
        self.conv4 = Conv2d(256, 512, kernel_size=3, stride=2)

        self.bn1 = BatchNorm2d(64)
        self.bn2 = BatchNorm2d(128)
        self.bn3 = BatchNorm2d(256)
        self.bn4 = BatchNorm2d(512)

        self.d_fc = Linear(512 +10, 256)

        self.merge_layer = Linear(256 , 1)

    def forward(self, input, label):
        network = self.conv1(input)
        network = leaky_relu(network, .2)
        network = self.bn1(network)

        network = self.conv2(network)
        network = leaky_relu(network, .2)
        network = self.bn2(network)

        network = self.conv3(network)
        network = leaky_relu(network, .2)
        network = self.bn3(network)

        network = self.conv4(network)
        network = leaky_relu(network, .2)
        network = self.bn4(network)

        network = network.view(-1, 512)
        network= torch.cat( [ network , label ], 1)

        network = self.d_fc(network)
        network = leaky_relu(network, .2)
        network = self.merge_layer(network)
        network = sigmoid(network)

        return network

'''
生成数据
'''
def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

def one_hot(target):
    y = torch.zeros(target.size()[0], 10)

    for i in range(target.size()[0]):
        y[i, target[i]] = 1

    return y

transform = Compose([
    Resize(config.IMAGE_SIZE),
    ToTensor()])

cifar = Cifar10DataSet(root='../data/cifar10/', train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(cifar, batch_size=BATCH_SIZE, shuffle=True)

'''
开始训练
'''
Net_D = Discriminator()
Net_G = Generator()
if GPU_NUMS > 1:
    Net_G.cuda()
    Net_D.cuda()
Net_D = DataParallel(Net_D)
Net_G = DataParallel(Net_G)

G_optimizer = Adam(Net_G.parameters(), lr=LR, betas=(0.5, 0.999))
D_optimizer = Adam(Net_G.parameters(), lr=LR, betas=(0.5, 0.999))

label_true = torch.ones(BATCH_SIZE)
label_false = torch.zeros(BATCH_SIZE)
label_true_var = Variable(label_true.cuda() if GPU_NUMS > 1 else label_true)
label_false_var = Variable(label_false.cuda() if GPU_NUMS > 1 else label_false)
proBar = ProgressBar(EPOCHS, len(train_loader), "D loss:%.3f; G loss:%.3f")
for epoch in range(EPOCHS):
    for image, label in train_loader:
        label = one_hot(label.long().squeeze())
        image_var = Variable(image.cuda() if GPU_NUMS > 1 else image)
        label_var = Variable(label.cuda() if GPU_NUMS > 1 else label)

        Noise_var = Variable(torch.randn(BATCH_SIZE, NOISE_DIM))
        '''
        训练判别器
        '''
        Net_D.zero_grad()
        D_real = Net_D(image_var, label_var)
        D_real_loss = BCELoss()(D_real, label_true_var)

        image_fake = Net_G(Noise_var, label_var)
        D_fake = Net_D(image_fake, label_var)
        D_fake_loss = BCELoss()(D_fake, label_false_var)

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_optimizer.step()

        '''
        训练生成器
        '''
        Net_G.zero_grad()
        Noise_var = Variable(torch.randn(BATCH_SIZE, NOISE_DIM))
        image_fake = Net_G(Noise_var,label_var)
        D_fake = Net_D(image_fake,label_var)

        G_loss = BCELoss()(D_fake, label_true_var)

        G_loss.backward()
        G_optimizer.step()

        proBar.show(D_loss.data[0], G_loss.data[0])
    Noise_var = Variable(torch.randn(BATCH_SIZE, NOISE_DIM))
    y = (torch.ones(BATCH_SIZE) * 7).long()
    y = one_hot(y)
    y = Variable(y.cuda() if GPU_NUMS > 1 else y)
    samples = Net_G(Noise_var,y)[:24]
    img = torchvision.utils.make_grid( samples.data)
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

    if not os.path.exists('out/'):
        os.makedirs('out/')

    plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close()

    torch.save(Net_G.state_dict(),"out/Net_G_%s.pth" % epoch)