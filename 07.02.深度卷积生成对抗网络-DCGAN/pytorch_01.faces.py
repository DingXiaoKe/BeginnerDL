# coding=utf-8
import torch as t
import torchvision as tv
from torch.utils.data import DataLoader
from torchvision.transforms import CenterCrop, ToTensor, Normalize,Compose,Resize
from torchvision.datasets import ImageFolder
from torch.nn import Module, Sequential, ConvTranspose2d, BatchNorm2d, ReLU,Tanh, Conv2d,\
    LeakyReLU, Sigmoid, BCELoss,DataParallel
from torch.optim import Adam, RMSprop, SGD
from torch.autograd import Variable

from lib.config.AnimeHeadConfig import AnimeHeadConfig
from lib.utils.progressbar.ProgressBar import ProgressBar

config = AnimeHeadConfig()
PHRASE = "TRAIN"
config.GPU_NUM = 2
class GeneratorNet(Module):
    def __init__(self):
        super(GeneratorNet, self).__init__()
        self.mainNetwork = Sequential(
            # 100,1,1 => 64*8,4,4
            ConvTranspose2d(config.NOISE_Z, config.GENERATOR_FEATURES_NUM * 8,
                            kernel_size=4, stride=1, padding=0, bias=False),
            BatchNorm2d(config.GENERATOR_FEATURES_NUM * 8),
            ReLU(True),

            # 64*8,4,4 => 64*4,8,8
            ConvTranspose2d(config.GENERATOR_FEATURES_NUM * 8, config.GENERATOR_FEATURES_NUM * 4,
                            kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(config.GENERATOR_FEATURES_NUM * 4),
            ReLU(True),

            # 64*4,8,8 => 64*2,16,16
            ConvTranspose2d(config.GENERATOR_FEATURES_NUM * 4, config.GENERATOR_FEATURES_NUM * 2,
                            kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(config.GENERATOR_FEATURES_NUM * 2),
            ReLU(True),

            # 64*2,16,16 => 64,32,32
            ConvTranspose2d(config.GENERATOR_FEATURES_NUM * 2, config.GENERATOR_FEATURES_NUM,
                            kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(config.GENERATOR_FEATURES_NUM),
            ReLU(True),

            # 64*2,32,32 => 3,96,96
            ConvTranspose2d(config.GENERATOR_FEATURES_NUM, 3, kernel_size=5,
                            stride=3, padding=1, bias=False),
            Tanh() # 3 * 96 * 96
        )

    def forward(self, input):
        return self.mainNetwork(input)

class DiscriminatorNet(Module):
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        self.mainNetwork = Sequential(
            # 3,96,96 => 64,32,32
            Conv2d(in_channels=3, out_channels=config.DISCRIMINATOR_FEATURES_NUM,
                   kernel_size=5, stride=3, padding=1, bias=False),
            LeakyReLU(0.2, inplace=True),

            #64,32,32 => 64*2,16,16
            Conv2d(in_channels=config.DISCRIMINATOR_FEATURES_NUM, out_channels=config.DISCRIMINATOR_FEATURES_NUM * 2,
                   kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(config.DISCRIMINATOR_FEATURES_NUM * 2),
            LeakyReLU(0.2, inplace=True),

            #64*2,16,16 => 64*4,8,8
            Conv2d(config.DISCRIMINATOR_FEATURES_NUM * 2, config.DISCRIMINATOR_FEATURES_NUM * 4,
                   kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(config.DISCRIMINATOR_FEATURES_NUM * 4),
            LeakyReLU(0.2, inplace=True),

            #64*4,8,8 => 64*8,4,4
            Conv2d(config.DISCRIMINATOR_FEATURES_NUM * 4, config.DISCRIMINATOR_FEATURES_NUM * 8,
                   kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm2d(config.DISCRIMINATOR_FEATURES_NUM * 8),
            LeakyReLU(0.2, inplace=True),

            Conv2d(config.DISCRIMINATOR_FEATURES_NUM * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            Sigmoid()
        )
    def forward(self, input):
        return self.mainNetwork(input).view(-1)

if PHRASE == "TRAIN":
    transforms = Compose([
        Resize(config.IMAGE_SIZE),
        CenterCrop(config.IMAGE_SIZE),
        ToTensor(),
        Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    dataset = ImageFolder(config.DATA_PATH, transform=transforms)
    dataLoader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE,
                            shuffle=True,num_workers=config.NUM_WORKERS_LOAD_IMAGE,
                            drop_last=True)
    netG, netD = DataParallel(GeneratorNet()), DataParallel(DiscriminatorNet())
    map_location = lambda storage, loc: storage

    optimizer_generator = Adam(netG.parameters(), config.LR_GENERATOR, betas=(config.BETA1, 0.999))
    optimizer_discriminator = Adam(netD.parameters(), config.LR_DISCRIMINATOR,betas=(config.BETA1, 0.999))

    criterion = BCELoss()

    true_labels = Variable(t.ones(config.BATCH_SIZE))
    fake_labels = Variable(t.zeros(config.BATCH_SIZE))
    fix_noises = Variable(t.randn(config.BATCH_SIZE,config.NOISE_Z,1,1))
    noises = Variable(t.randn(config.BATCH_SIZE,config.NOISE_Z,1,1))

    # errord_meter = AverageValueMeter()
    # errorg_meter = AverageValueMeter()

    if config.GPU_NUM > 1:
        netD.cuda()
        netG.cuda()
        criterion.cuda()
        true_labels,fake_labels = true_labels.cuda(), fake_labels.cuda()
        fix_noises,noises = fix_noises.cuda(),noises.cuda()

    epochs = range(config.EPOCH_NUM)
    proBar = ProgressBar(config.EPOCH_NUM, len(dataLoader), "D Loss:%.3f;G Loss:%.3f")
    for epoch in iter(epochs):
        for ii,(img,_) in enumerate(dataLoader):
            real_img = Variable(img)
            if config.GPU_NUM > 1:
                real_img=real_img.cuda()
            if ii % config.D_EVERY==0:
                # 训练判别器
                optimizer_discriminator.zero_grad()
                ## 尽可能的把真图片判别为正确
                output = netD(real_img)
                error_d_real = criterion(output,true_labels)
                error_d_real.backward()

                ## 尽可能把假图片判别为错误
                noises.data.copy_(t.randn(config.BATCH_SIZE,config.NOISE_Z,1,1))
                fake_img = netG(noises).detach() # 根据噪声生成假图
                output = netD(fake_img)
                error_d_fake = criterion(output,fake_labels)
                error_d_fake.backward()
                optimizer_discriminator.step()

                error_d = error_d_fake + error_d_real

                # errord_meter.add(error_d.data[0])

            if ii % config.G_EVERY==0:
                # 训练生成器
                optimizer_generator.zero_grad()
                noises.data.copy_(t.randn(config.BATCH_SIZE,config.NOISE_Z,1,1))
                fake_img = netG(noises)
                output = netD(fake_img)
                error_g = criterion(output,true_labels)
                error_g.backward()
                optimizer_generator.step()
                # errorg_meter.add(error_g.data[0])

            proBar.show(error_d.data[0], error_g.data[0])
            fix_fake_imgs = netG(fix_noises)
            # if opt.vis and ii%opt.plot_every == opt.plot_every-1:
            #     ## 可视化
            #     if os.path.exists(opt.debug_file):
            #         ipdb.set_trace()
            #     fix_fake_imgs = netg(fix_noises)
            #     vis.images(fix_fake_imgs.data.cpu().numpy()[:64]*0.5+0.5,win='fixfake')
            #     vis.images(real_img.data.cpu().numpy()[:64]*0.5+0.5,win='real')
            #     vis.plot('errord',errord_meter.value()[0])
            #     vis.plot('errorg',errorg_meter.value()[0])

        if epoch % config.DECAY_EVERY == 0:
            # 保存模型、图片
            tv.utils.save_image(fix_fake_imgs.data[:64],'%s/%s.png' %(config.SAVE_PATH,epoch),normalize=True,range=(-1,1))
            t.save(netD.state_dict(),'checkpoints/netd_%s.pth' %epoch)
            t.save(netG.state_dict(),'checkpoints/netg_%s.pth' %epoch)
            # errord_meter.reset()
            # errorg_meter.reset()
            optimizer_g = t.optim.Adam(netG.parameters(),config.LR_GENERATOR,betas=(config.BETA1, 0.999))
            optimizer_d = t.optim.Adam(netD.parameters(),config.LR_DISCRIMINATOR,betas=(config.BETA1, 0.999))

