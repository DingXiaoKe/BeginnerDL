# coding=utf-8
import torch
import torchvision
import numpy as np
import os

from torch.nn import Module, Sequential, ConvTranspose2d, BatchNorm2d, ReLU, Tanh, Conv2d, LeakyReLU, DataParallel, BCELoss,\
    Linear,Sigmoid
from torch.nn.init import normal, constant
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop, Normalize
from torch.optim import Adam
from torch.autograd import Variable
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from torchvision.datasets import ImageFolder
from lib.datareader.pytorch.Faces_partition import FacesGenderDataSet
from lib.utils.progressbar.ProgressBar import ProgressBar

BATCH_SIZE = 128
GPU_NUMS = 2
EPOCHS = 20
IMAGE_SIZE = 96
IMAGE_CHANNEL = 3
NOISE_DIM = 100
LR = 0.0002
BETAS = (0.5, 0.999)

'''
生成器
input = [BATCH_SIZE, NOISE_DIM]
label = [BATCH_SIZE, 10]
'''
class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.input_conv1 = ConvTranspose2d(NOISE_DIM, 512, kernel_size=4, stride=1, padding=0)
        normal(self.input_conv1.weight, mean=0.0, std=0.02)
        constant(self.input_conv1.bias, 0.0)

        self.label_conv1 = ConvTranspose2d(2, 512, kernel_size=4, stride=1, padding=0)
        normal(self.label_conv1.weight, mean=0.0, std=0.02)
        constant(self.label_conv1.bias, 0.0)

        self.input_bn1 = BatchNorm2d(512)
        self.label_bn1 = BatchNorm2d(512)

        self.merge_conv1 = ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        normal(self.merge_conv1.weight, mean=0.0, std=0.02)
        constant(self.merge_conv1.bias, 0.0)
        self.merge_bn1 = BatchNorm2d(512)

        self.merge_conv2 = ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        normal(self.merge_conv2.weight, mean=0.0, std=0.02)
        constant(self.merge_conv2.bias, 0.0)
        self.merge_bn2 = BatchNorm2d(256)

        self.merge_conv3 = ConvTranspose2d(256, 128, kernel_size=5, stride=3, padding=1)
        normal(self.merge_conv3.weight, mean=0.0, std=0.02)
        constant(self.merge_conv3.bias, 0.0)
        self.merge_bn3 = BatchNorm2d(128)

        self.full_conv1 = ConvTranspose2d(128,  3, kernel_size=4, stride=2, padding=1)
        normal(self.full_conv1.weight, mean=0.0, std=0.02)
        constant(self.full_conv1.bias, 0.0)
    def forward(self, z, c):
        network1 = self.input_conv1(z)
        network1 = ReLU()(network1)

        network2 = self.label_conv1(c)
        network2 = ReLU()(network2)

        network  = torch.cat([network1, network2], dim=1)

        network  = self.merge_conv1(network)
        network  = self.merge_bn1(network)
        network  = ReLU()(network)

        network  = self.merge_conv2(network)
        network  = self.merge_bn2(network)
        network  = ReLU()(network)

        network  = self.merge_conv3(network)
        network  = self.merge_bn3(network)
        network  = ReLU()(network)

        network  = self.full_conv1(network)
        network  = Tanh()(network)
        return network
# class Generator(Module):
#     def __init__(self, feature_num):
#         super(Generator, self).__init__()
#         self.feature_num = feature_num
#         self.gc_full = Linear(NOISE_DIM + 10, NOISE_DIM + 10)
#         self.deconv1 = ConvTranspose2d(NOISE_DIM + 10, feature_num * 8,
#                                      kernel_size=4, stride=1, padding=0, bias=False)
#         self.bn1 = BatchNorm2d(feature_num * 8)
#
#         self.deconv2 = ConvTranspose2d(feature_num * 8, feature_num * 4,
#                                        kernel_size=4, stride=2, padding=1, bias=False)
#         self.bn2 = BatchNorm2d(feature_num * 4)
#
#         self.deconv3 = ConvTranspose2d(feature_num * 4, feature_num * 2,
#                                      kernel_size=4, stride=2, padding=1, bias=False)
#         self.bn3 = BatchNorm2d(feature_num * 2)
#
#         self.deconv4 = ConvTranspose2d(feature_num * 2, feature_num,
#                                      kernel_size=4, stride=2, padding=1, bias=False)
#         self.bn4 = BatchNorm2d(feature_num)
#
#         self.deconv5 = ConvTranspose2d(feature_num, IMAGE_CHANNEL, kernel_size=5,
#                                        stride=3, padding=1, bias=False)
#     def forward(self, input, label):
#         network = torch.cat([input, label], dim=1)
#         network = self.gc_full(network)
#         network = network.view(-1, NOISE_DIM + 10, 1, 1)
#         network = self.deconv1(network)
#         network = self.bn1(network)
#         network = ReLU()(network)
#
#         network = self.deconv2(network)
#         network = self.bn2(network)
#         network = ReLU()(network)
#
#         network = self.deconv3(network)
#         network = self.bn3(network)
#         network = ReLU()(network)
#
#         network = self.deconv4(network)
#         network = self.bn4(network)
#         network = ReLU()(network)
#
#         network = self.deconv5(network)
#         network = Tanh()(network)
#         return network
'''
判别器
input [BATCH_SIZE, 3, 96, 96]
'''
class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.input_conv = Conv2d(in_channels=3, out_channels=512, kernel_size=4, stride=2, padding=1)
        normal(self.input_conv.weight, mean=0.0, std=0.02)
        constant(self.input_conv.bias, 0.0)

        self.label_conv = Conv2d(in_channels=2, out_channels=512, kernel_size=4, stride=2, padding=1)
        normal(self.input_conv.weight, mean=0.0, std=0.02)
        constant(self.input_conv.bias, 0.0)

        self.merge_conv1 = Conv2d(in_channels=1024, out_channels=512, kernel_size=4, stride=3, padding=1)
        normal(self.merge_conv1.weight, mean=0.0, std=0.02)
        constant(self.merge_conv1.bias, 0.0)
        self.merge_bn1   = BatchNorm2d(512)

        self.merge_conv2 = Conv2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        normal(self.merge_conv2.weight, mean=0.0, std=0.02)
        constant(self.merge_conv2.bias, 0.0)
        self.merge_bn2   = BatchNorm2d(256)

        self.merge_conv3 = Conv2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        normal(self.merge_conv3.weight, mean=0.0, std=0.02)
        constant(self.merge_conv3.bias, 0.0)
        self.merge_bn3 = BatchNorm2d(128)

        self.full_conv1 = Conv2d(in_channels=128, out_channels=1,kernel_size=4, stride=4, padding=0)
        normal(self.full_conv1.weight, mean=0.0, std=0.02)
        constant(self.full_conv1.bias, 0.0)
        # for i in range(len(num_filters)):
        #     # Convolutional layer
        #     if i == 0:
        #         # For input
        #         input_conv = torch.nn.Conv2d(input_dim, int(num_filters[i]/2), kernel_size=4, stride=2, padding=1)
        #         self.hidden_layer1.add_module('input_conv', input_conv)
        #
        #         # Initializer
        #         torch.nn.init.normal(input_conv.weight, mean=0.0, std=0.02)
        #         torch.nn.init.constant(input_conv.bias, 0.0)
        #
        #         # Activation
        #         self.hidden_layer1.add_module('input_act', torch.nn.LeakyReLU(0.2))
        #
        #         # For label
        #         label_conv = torch.nn.Conv2d(label_dim, int(num_filters[i]/2), kernel_size=4, stride=2, padding=1)
        #         self.hidden_layer2.add_module('label_conv', label_conv)
        #
        #         # Initializer
        #         torch.nn.init.normal(label_conv.weight, mean=0.0, std=0.02)
        #         torch.nn.init.constant(label_conv.bias, 0.0)
        #
        #         # Activation
        #         self.hidden_layer2.add_module('label_act', torch.nn.LeakyReLU(0.2))
        #     else:
        #         conv = torch.nn.Conv2d(num_filters[i-1], num_filters[i], kernel_size=4, stride=2, padding=1)
        #
        #         conv_name = 'conv' + str(i + 1)
        #         self.hidden_layer.add_module(conv_name, conv)
        #
        #         # Initializer
        #         torch.nn.init.normal(conv.weight, mean=0.0, std=0.02)
        #         torch.nn.init.constant(conv.bias, 0.0)
        #
        #         # Batch normalization
        #         bn_name = 'bn' + str(i + 1)
        #         self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))
        #
        #         # Activation
        #         act_name = 'act' + str(i + 1)
        #         self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))
        #
        # # Output layer
        # self.output_layer = torch.nn.Sequential()
        # # Convolutional layer
        # out = torch.nn.Conv2d(num_filters[i], output_dim, kernel_size=4, stride=1, padding=0)
        # self.output_layer.add_module('out', out)
        # # Initializer
        # torch.nn.init.normal(out.weight, mean=0.0, std=0.02)
        # torch.nn.init.constant(out.bias, 0.0)
        # # Activation
        # self.output_layer.add_module('act', torch.nn.Sigmoid())

    def forward(self, z, c):
        network1 = self.input_conv(z)
        network1 = LeakyReLU(negative_slope=0.2)(network1)

        network2 = self.label_conv(c)
        network2 = LeakyReLU(negative_slope=0.2)(network2)

        network  = torch.cat([network1, network2], dim=1)

        network  = self.merge_conv1(network)
        network  = self.merge_bn1(network)
        network  = LeakyReLU(negative_slope=0.2)(network)

        network  = self.merge_conv2(network)
        network  = self.merge_bn2(network)
        network  = LeakyReLU(negative_slope=0.2)(network)

        network  = self.merge_conv3(network)
        network  = self.merge_bn3(network)
        network  = LeakyReLU(negative_slope=0.2)(network)

        network  = self.full_conv1(network)
        network  = Sigmoid()(network)
        return network
        # c = c.view(-1, 10, 1, 1)

        # h = self.hidden_layer(x)
        # out = self.output_layer(h)
        # return out
# class Discriminator(Module):
    # def __init__(self, features_num=64):
    #     super(Discriminator, self).__init__()
    #     self.features_num = features_num
    #     self.input_conv1 = Conv2d(in_channels=IMAGE_CHANNEL, out_channels=features_num,
    #                               kernel_size=5, stride=3, padding=1, bias=False)
    #     self.input_conv2 = Conv2d(in_channels=features_num, out_channels=features_num * 2,
    #                               kernel_size=4, stride=2, padding=1, bias=False)
    #     self.input_conv3 = Conv2d(features_num * 2, features_num * 4,
    #                               kernel_size=4, stride=2, padding=1, bias=False)
    #     self.input_bn3 =  BatchNorm2d(features_num * 4)
    #     self.input_conv4 = Conv2d(features_num * 4, features_num * 8,
    #                               kernel_size=4, stride=2, padding=1, bias=False)
    #     self.input_bn4 =  BatchNorm2d(features_num * 8)
    #     self.input_conv5 = Conv2d(features_num * 8, features_num * 8, kernel_size=4, stride=1, padding=0, bias=False)
    #
    #     self.cat_conv1 = Linear(512 +10, 256)
    #     self.cat_merge = Linear(256 , 1)
    # def forward(self, input, label):
    #     network_1 = self.input_conv1(input)
    #     network_1 = LeakyReLU(negative_slope=0.2, inplace=True)(network_1)
    #     network_1 = self.input_conv2(network_1)
    #     network_1 = LeakyReLU(negative_slope=0.2, inplace=True)(network_1)
    #     network_1 = self.input_conv3(network_1)
    #     network_1 = self.input_bn3(network_1)
    #     network_1 = LeakyReLU(negative_slope=0.2, inplace=True)(network_1)
    #     network_1 = self.input_conv4(network_1)
    #     network_1 = self.input_bn4(network_1)
    #     network_1 = LeakyReLU(negative_slope=0.2, inplace=True)(network_1)
    #     network_1 = self.input_conv5(network_1)
    #     network_1 = network_1.view(-1, self.features_num * 8)
    #     network= torch.cat( [ network_1 , label ], 1)
    #     network = self.cat_conv1(network)
    #     network = LeakyReLU(negative_slope=0.2, inplace=True)(network)
    #     network = self.cat_merge(network)
    #     network = Sigmoid()(network)
    #
    #     return network
'''
生成网络
'''
Net_G = Generator()
Net_D = Discriminator()
Net_G = DataParallel(Net_G)
Net_D = DataParallel(Net_D)
if GPU_NUMS > 1:
    Net_D.cuda()
    Net_G.cuda()
G_optimizer = Adam(Net_G.parameters(), lr=LR, betas=BETAS)
D_optimizer = Adam(Net_D.parameters(), lr=LR, betas=BETAS)

'''
数据读入与预处理
'''
transforms = Compose([
    Resize(IMAGE_SIZE),
    CenterCrop(IMAGE_SIZE),
    ToTensor(),
    Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

dataset = ImageFolder(root='../ganData/face_gender/', transform=transforms)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def one_hot(target):
    y = torch.zeros(target.size()[0], 10)

    for i in range(target.size()[0]):
        y[i, target[i]] = 1

    return y
'''
开始训练
'''
onehot = torch.zeros(2, 2)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1]).view(2, 1), 1).view(2, 2, 1, 1)

fill = torch.zeros([2, 2, IMAGE_SIZE, IMAGE_SIZE])
for i in range(2):
    fill[i, i, :, :] = 1


proBar = ProgressBar(EPOCHS, len(train_loader), "D Loss:%.3f;G Loss:%.3f")
for epoch in range(1, EPOCHS + 1):
    if epoch == 5 or epoch == 10:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10

    for image, label in train_loader:
        mini_batch = image.size()[0]
        label_true = torch.ones(mini_batch)
        label_false = torch.zeros(mini_batch)
        label_true_var = Variable(label_true.cuda() if GPU_NUMS > 1 else label_true)
        label_false_var = Variable(label_false.cuda() if GPU_NUMS > 1 else label_false)

        # label = one_hot(label.long().squeeze())
        label_onehot = fill[label]
        image_var = Variable(image.cuda() if GPU_NUMS > 1 else image)
        label_var = Variable(label_onehot.cuda() if GPU_NUMS > 1 else label_onehot)



        '''hidden_layers
        训练判别器
        '''
        Net_D.zero_grad()

        # 真值损失
        D_real = Net_D(image_var, label_var)
        D_real_loss = BCELoss()(D_real, label_true_var)

        # 假值损失
        img_fake = torch.randn((mini_batch, NOISE_DIM)).view(-1, NOISE_DIM, 1, 1)
        label_fake = (torch.rand(mini_batch, 1) * 2).type(torch.LongTensor).squeeze()
        img_fake_var = Variable(img_fake.cuda() if GPU_NUMS > 1 else img_fake)
        label_fake_G_var = Variable(onehot[label_fake].cuda() if GPU_NUMS > 1 else onehot[label_fake])
        label_fake_D_var = Variable(fill[label_fake].cuda() if GPU_NUMS > 1 else fill[label_fake])

        image_fake = Net_G(img_fake_var, label_fake_G_var)
        D_fake = Net_D(image_fake, label_fake_D_var)
        D_fake_loss = BCELoss()(D_fake, label_false_var)

        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        D_optimizer.step()

        '''
        训练生成器
        '''
        Net_G.zero_grad()
        img_fake = torch.randn((mini_batch, NOISE_DIM)).view(-1, NOISE_DIM, 1, 1)
        label_fake = (torch.rand(mini_batch, 1) * 2).type(torch.LongTensor).squeeze()
        img_fake_var = Variable(img_fake.cuda() if GPU_NUMS > 1 else img_fake)
        label_fake_G_var = Variable(onehot[label_fake].cuda() if GPU_NUMS > 1 else onehot[label_fake])
        label_fake_D_var = Variable(fill[label_fake].cuda() if GPU_NUMS > 1 else fill[label_fake])
        image_fake = Net_G(img_fake_var,label_fake_G_var)
        D_fake = Net_D(image_fake,label_fake_D_var)

        G_loss = BCELoss()(D_fake, label_true_var)

        G_loss.backward()
        G_optimizer.step()

        proBar.show(D_loss.data[0], G_loss.data[0])

    temp_size = 32
    img_fake = torch.randn((temp_size, NOISE_DIM)).view(-1, NOISE_DIM, 1, 1)
    img_fake_var = Variable(img_fake.cuda() if GPU_NUMS > 1 else img_fake)
    y = torch.cat([torch.zeros(8), torch.ones(8), torch.zeros(8), torch.ones(8)], 0).type(torch.LongTensor).squeeze()
    y = onehot[y]
    y_var = Variable(y.cuda() if GPU_NUMS > 1 else y)
    samples = Net_G(img_fake_var,y_var)

    if not os.path.exists('out/'):
        os.makedirs('out/')

    # plt.savefig('out/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    # plt.close()
    torchvision.utils.save_image(samples.data,'out/%s.png' %(epoch),normalize=True,range=(-1,1))
    torch.save(Net_G.state_dict(),"out/Net_G_%s.pth" % epoch)


