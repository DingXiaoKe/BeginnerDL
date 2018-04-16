# coding=utf-8
import torch
import numpy as np

from os.path import join
from torch.utils.data import DataLoader
from torch.nn import Module, BatchNorm2d, InstanceNorm2d, Conv2d, ReLU, Dropout, Sequential, ConvTranspose2d, Tanh, DataParallel,\
    LeakyReLU, Sigmoid, MSELoss, BCELoss, L1Loss
from torch.optim import Adam
from torch.autograd import Variable
from lib.datareader.pytorch.pictures_transfer import DataSetFromFolderForPicTransfer
from lib.utils.progressbar.ProgressBar import ProgressBar
from lib.utils.utils.picture_transfer_show import plot_test_result

GPU_NUMS= 2
EPOCHS = 200
LR = 0.0002
BETA = 0.5
BATCH_SIZE = 1
IMAGE_CHANNEL = 3
IMAGE_SIZE = 3
OUTPUT_CHANNEL = 3
'''
生成网络
'''
class Generator(Module):
    def __init__(self, input_nc = 3, output_nc = 3, ngf = 64, norm='batch', use_dropout=False):
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.feature_nums = ngf
        self.norm = norm
        self.use_dropout = use_dropout
        self.n_blocks = 9
        self.norm_layer = BatchNorm2d if self.norm == 'batch' else InstanceNorm2d

        model = [Conv2d(self.input_nc, self.feature_nums, kernel_size=7, padding=3),
                 self.norm_layer(self.feature_nums, affine=True),
                 ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [Conv2d(self.feature_nums * mult, self.feature_nums * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      self.norm_layer(self.feature_nums * mult * 2, affine=True),
                      ReLU(True)]

        mult = 2**n_downsampling # 4
        for i in range(self.n_blocks): # 9
            model += [ResnetBlock(self.feature_nums * mult, 'zero', norm_layer=self.norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [ConvTranspose2d(self.feature_nums * mult, int(self.feature_nums * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      self.norm_layer(int(self.feature_nums * mult / 2), affine=True),
                      ReLU(True)]

        model += [Conv2d(self.feature_nums, self.output_nc, kernel_size=7, padding=3)]
        model += [Tanh()]

        self.model = Sequential(*model)

    def forward(self, input):
        return self.model(input)

class ResnetBlock(Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.dim = dim
        self.padding_type = padding_type
        self.norm_layer = norm_layer
        self.use_dropout = use_dropout
        self.conv_block = self._build_conv_block()

    def forward(self, x):
        return x + self.conv_block(x)

    def _build_conv_block(self):
        conv_block = []
        p = 1
        conv_block += [Conv2d(self.dim, self.dim, kernel_size=3, padding=p),
                       self.norm_layer(self.dim, affine=True),
                       ReLU(True)]
        if self.use_dropout:
            conv_block += [Dropout(float=0.5)]
        conv_block += [Conv2d(self.dim, self.dim, kernel_size=3, padding=p),
                       self.norm_layer(self.dim, affine=True)]
        return Sequential(*conv_block)

'''
判别网络 PatchGAN
'''
class Discriminator(Module):
    def __init__(self, input_nc = 6, ndf = 64, n_layers = 3, norm = 'batch', use_sigmoid=False):
        super(Discriminator, self).__init__()
        kw = 4
        padw = int(np.ceil((kw-1)/2))
        self.norm_layer = BatchNorm2d if norm == 'batch' else InstanceNorm2d
        sequence = [
            Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2,padding=padw),
                self.norm_layer(ndf * nf_mult,affine=True),
                LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1,
                      padding=padw),
            self.norm_layer(ndf * nf_mult,affine=True),
            LeakyReLU(0.2, True)
        ]

        sequence += [Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [Sigmoid()]

        self.model = Sequential(*sequence)
    def forward(self, input):
        return self.model(input)

'''
定义损失
'''
class GANLoss(Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = MSELoss()
        else:
            self.loss = BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor.cuda() if GPU_NUMS > 1 else target_tensor)
'''
创建网络
'''
Net_G = DataParallel(Generator())
Net_D = DataParallel(Discriminator())
lossGAN = GANLoss()
lossL1 = L1Loss()
lossMSE = MSELoss()

optimizer_G = Adam(Net_G.parameters(), lr=LR, betas=(BETA, 0.999))
optimizer_D = Adam(Net_D.parameters(), lr=LR, betas=(BETA, 0.999))
'''
生成数据
'''
train_set = DataSetFromFolderForPicTransfer(join("../ganData/facades_fixed", "train"))
test_set  = DataSetFromFolderForPicTransfer(join("../ganData/facades_fixed", "test"))
train_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader  = DataLoader(dataset=test_set, num_workers=2, batch_size=BATCH_SIZE, shuffle=True)
test_input, test_target = test_data_loader.__iter__().__next__()

real_a = torch.FloatTensor(BATCH_SIZE, IMAGE_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)
real_b = torch.FloatTensor(BATCH_SIZE, OUTPUT_CHANNEL, IMAGE_SIZE, IMAGE_SIZE)

if GPU_NUMS > 1:
    Net_G = Net_G.cuda()
    Net_D = Net_D.cuda()
    lossGAN = lossGAN.cuda()
    lossL1 = lossL1.cuda()
    lossMSE = lossMSE.cuda()

real_a = Variable(real_a.cuda() if GPU_NUMS > 1 else real_a)
real_b = Variable(real_b.cuda() if GPU_NUMS > 1 else real_b)

bar = ProgressBar(EPOCHS, len(train_data_loader), "D loss:%.3f;G loss:%.3f")
for epoch in range(EPOCHS):
    for iteration, batch in enumerate(train_data_loader, 1):
        real_a_cpu, real_b_cpu = batch[0], batch[1]
        real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
        fake_b = Net_G(real_a)

        optimizer_D.zero_grad()

        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = Net_D.forward(fake_ab.detach())
        loss_d_fake = lossGAN(pred_fake, False)

        # train with real
        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = Net_D.forward(real_ab)
        loss_d_real = lossGAN(pred_real, True)

        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5
        loss_d.backward()
        optimizer_D.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizer_G.zero_grad()
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = Net_D.forward(fake_ab)
        loss_g_gan = lossGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = lossL1(fake_b, real_b) * 10
        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()
        optimizer_G.step()

        bar.show(loss_d.data[0], loss_g.data[0])

    gen_image = Net_G(Variable(test_input.cuda() if GPU_NUMS > 1 else test_input))
    gen_image = gen_image.cpu().data
    plot_test_result(test_input, test_target, gen_image, epoch, save=True, save_dir='output/')
    torch.save(Net_G.state_dict(),"output/Net_G_%s.pth" % epoch)


