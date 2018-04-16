# coding=utf-8
import torch
import os

from torch.nn import Module, DataParallel, Conv2d, LeakyReLU, BatchNorm2d, Dropout, ConvTranspose2d, ReLU, BCELoss, L1Loss
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from lib.datareader.pytorch.pictures_transfer import DataSetFromFolderForPicTransfer
from lib.utils.progressbar.ProgressBar import ProgressBar
from lib.utils.utils.picture_transfer_show import plot_test_result
GPU_NUMS = 2
EPOCHS = 200
IMAGE_CHANNEL = 3
IMAGE_SIZE = 256
OUT_CHANNEL = 3
BATCH_SIZE = 1
'''
声明生成器
'''
class Generator(Module):
    def __init__(self, num_filter=64):
        super(Generator, self).__init__()
        self.num_filters = num_filter
        self.conv1 = ConvBlock(IMAGE_CHANNEL, self.num_filters, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(self.num_filters, self.num_filters * 2)
        self.conv3 = ConvBlock(self.num_filters * 2, self.num_filters * 4)
        self.conv4 = ConvBlock(self.num_filters * 4, self.num_filters * 8)
        self.conv5 = ConvBlock(self.num_filters * 8, self.num_filters * 8)
        self.conv6 = ConvBlock(self.num_filters * 8, self.num_filters * 8)
        self.conv7 = ConvBlock(self.num_filters * 8, self.num_filters * 8)
        self.conv8 = ConvBlock(self.num_filters * 8, self.num_filters * 8, batch_norm=False)

        self.deconv1 = DeconvBlock(num_filter * 8, num_filter * 8, dropout=True)
        self.deconv2 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv3 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv4 = DeconvBlock(num_filter * 8 * 2, num_filter * 8)
        self.deconv5 = DeconvBlock(num_filter * 8 * 2, num_filter * 4)
        self.deconv6 = DeconvBlock(num_filter * 4 * 2, num_filter * 2)
        self.deconv7 = DeconvBlock(num_filter * 2 * 2, num_filter)
        self.deconv8 = DeconvBlock(num_filter * 2, OUT_CHANNEL, batch_norm=False)
    def forward(self, input):
        enc1 = self.conv1(input)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        enc6 = self.conv6(enc5)
        enc7 = self.conv7(enc6)
        enc8 = self.conv8(enc7)
        # Decoder with skip-connections
        dec1 = self.deconv1(enc8)
        dec1 = torch.cat([dec1, enc7], 1)
        dec2 = self.deconv2(dec1)
        dec2 = torch.cat([dec2, enc6], 1)
        dec3 = self.deconv3(dec2)
        dec3 = torch.cat([dec3, enc5], 1)
        dec4 = self.deconv4(dec3)
        dec4 = torch.cat([dec4, enc4], 1)
        dec5 = self.deconv5(dec4)
        dec5 = torch.cat([dec5, enc3], 1)
        dec6 = self.deconv6(dec5)
        dec6 = torch.cat([dec6, enc2], 1)
        dec7 = self.deconv7(dec6)
        dec7 = torch.cat([dec7, enc1], 1)
        dec8 = self.deconv8(dec7)
        out = torch.nn.Tanh()(dec8)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                torch.nn.init.normal(m.deconv.weight, mean, std)

class ConvBlock(Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, strides=2, padding=1, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=strides, padding=padding)
        self.activation = activation
        self.lrelu = LeakyReLU(negative_slope=0.2, inplace=True)
        self.batch_norm = batch_norm
        self.bn = BatchNorm2d(out_channel)

    def forward(self, input):
        out = self.conv(input)
        if self.activation:
            out = self.lrelu(out)

        if self.batch_norm:
            out = self.bn(out)

        return out

class DeconvBlock(Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=True, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.bn = BatchNorm2d(output_size)
        self.drop = Dropout(0.5)
        self.relu = ReLU(True)
        self.batch_norm = batch_norm
        self.dropout = dropout

    def forward(self, x):
        out = self.deconv(x)
        out = self.relu(out)

        if self.batch_norm:
            out = self.bn(out)

        if self.dropout:
            out = self.drop(out)

        return out
'''
声明判别器
'''
class Discriminator(Module):
    def __init__(self, input_dim=6, num_filter=64, output_dim=1):
        super(Discriminator, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8, strides=1)
        self.conv5 = ConvBlock(num_filter * 8, output_dim, strides=1, batch_norm=False)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = torch.nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)
'''
构造网络
'''
Net_G = Generator()
Net_D = Discriminator()
Net_G.normal_weight_init(mean=0.0, std=0.02)
Net_D.normal_weight_init(mean=0.0, std=0.02)

Net_G = DataParallel(Net_G)
Net_D = DataParallel(Net_D)

BCE_loss = BCELoss()
L1_loss = L1Loss()

if GPU_NUMS > 1:
    Net_D.cuda()
    Net_G.cuda()
    BCE_loss = BCE_loss.cuda()
    L1_loss = L1_loss.cuda()
G_optimizer = Adam(Net_G.parameters(), lr=0.0002, betas=(0.5, 0.999))
D_optimizer = Adam(Net_D.parameters(), lr=0.0002, betas=(0.5, 0.999))

'''
读入数据
'''
if not os.path.exists("output"):
    os.mkdir("output")

train_set = DataSetFromFolderForPicTransfer(os.path.join("/data/facades_fixed", "train"))
test_set  = DataSetFromFolderForPicTransfer(os.path.join("/data/facades_fixed", "test"))
train_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=BATCH_SIZE, shuffle=True)
test_data_loader  = DataLoader(dataset=test_set, num_workers=2, batch_size=BATCH_SIZE, shuffle=True)

test_input, test_target = test_data_loader.__iter__().__next__()
'''
开始训练
'''
bar = ProgressBar(EPOCHS, len(train_data_loader), "D loss:%.3f;G loss:%.3f")
for epoch in range(EPOCHS):
    for i, (input, target) in enumerate(train_data_loader):
        x_ = Variable(input.cuda() if GPU_NUMS > 1 else input)
        y_ = Variable(target.cuda() if GPU_NUMS > 1 else target)

        # Train discriminator with real data
        D_real_decision = Net_D(x_, y_).squeeze()
        real_ = Variable(torch.ones(D_real_decision.size()).cuda() if GPU_NUMS > 1 else torch.ones(D_real_decision.size()))
        D_real_loss = BCE_loss(D_real_decision, real_)

        # Train discriminator with fake data
        gen_image = Net_G(x_)
        D_fake_decision = Net_D(x_, gen_image).squeeze()
        fake_ = Variable(torch.zeros(D_fake_decision.size()).cuda() if GPU_NUMS > 1 else torch.zeros(D_fake_decision.size()))
        D_fake_loss = BCE_loss(D_fake_decision, fake_)

        # Back propagation
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        Net_D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        gen_image = Net_G(x_)
        D_fake_decision = Net_D(x_, gen_image).squeeze()
        G_fake_loss = BCE_loss(D_fake_decision, real_)

        # L1 loss
        l1_loss = 100 * L1_loss(gen_image, y_)

        # Back propagation
        G_loss = G_fake_loss + l1_loss
        Net_G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        bar.show(D_loss.data[0], G_loss.data[0])

    gen_image = Net_G(Variable(test_input.cuda() if GPU_NUMS > 1 else test_input))
    gen_image = gen_image.cpu().data
    plot_test_result(test_input, test_target, gen_image, epoch, save=True, save_dir='output/')

torch.save(Net_G.state_dict(),"output/Net_G_20.pth")