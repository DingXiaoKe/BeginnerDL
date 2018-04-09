# coding=utf-8
import torch
import itertools

from torchvision.transforms import Compose,Scale,ToTensor, Normalize
from torch.nn import Module, ConvTranspose2d, BatchNorm2d, Conv2d,BCELoss,DataParallel
from torch.nn.functional import relu, leaky_relu, tanh, sigmoid
from torchvision.datasets import MNIST
from torch.optim import Adam
from torch.autograd import Variable
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from lib.datareader.pytorch.MNIST import MNISTDataSet
from lib.utils.progressbar.ProgressBar import ProgressBar

GPU_NUMS = 1
# G(z)
'''
定义Generator,
输入为input和label
input = [?,100,1,1]
label = [?, 10,1,1]

input => [?,100,1,1] => [?,256,4,4]
label => [?,10,1,1] => [?,256,4,4]

然后合并在一起变为[?,512,4,4]
然后继续扩展=>[?,256,8,8] => [?,128,16,16] => [128,1,32,32]
'''
class Generator(Module):
    def __init__(self, depth):
        super(Generator, self).__init__()
        # ?,100,1,1 => ?,256,4,4
        self.deconv1_1    = ConvTranspose2d(in_channels=100, out_channels=depth * 2, kernel_size=4, stride=1, padding=0)
        self.deconv1_1_bn = BatchNorm2d(depth * 2)
        self.deconv1_2    = ConvTranspose2d(10, depth*2, 4, 1, 0)
        self.deconv1_2_bn = BatchNorm2d(depth*2)
        self.deconv2      = ConvTranspose2d(depth*4, depth*2, 4, 2, 1)
        self.deconv2_bn   = BatchNorm2d(depth*2)
        self.deconv3      = ConvTranspose2d(depth*2, depth, 4, 2, 1)
        self.deconv3_bn   = BatchNorm2d(depth)
        self.deconv4      = ConvTranspose2d(depth, 1, 4, 2, 1)
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, label):
        network = self.deconv1_1(input)
        network = self.deconv1_1_bn(network)
        network = relu(network)

        network_branch = self.deconv1_2(label)
        network_branch = self.deconv1_2_bn(network_branch)
        network_branch = relu(network_branch)

        network = torch.cat([network, network_branch], dim=1)

        network = self.deconv2(network)
        network = self.deconv2_bn(network)
        network = relu(network)

        network = self.deconv3(network)
        network = self.deconv3_bn(network)
        network = relu(network)

        network = self.deconv4(network)
        network = tanh(network)

        return network

'''
定义Discriminator
输入为input和label
input = [?,1,32,32]
label = [?, 10,32,32]

input => [?,100,32,32] => [?,64,4,4]
label => [?,10,32,32] => [?,64,4,4]

然后合并在一起变为[?,128,16,16]
然后继续扩展=>[?,256,8,8] => [?,512,4,4] => [128,1,1,1]
'''
class Discriminator(Module):
    def __init__(self, depth):
        super(Discriminator, self).__init__()
        self.conv1_1  = Conv2d(1, int(depth/2), 4, 2, 1)
        self.conv1_2  = Conv2d(10, int(depth/2), 4, 2, 1)
        self.conv2    = Conv2d(depth, depth*2, 4, 2, 1)
        self.conv2_bn = BatchNorm2d(depth*2)
        self.conv3    = Conv2d(depth*2, depth*4, 4, 2, 1)
        self.conv3_bn = BatchNorm2d(depth*4)
        self.conv4    = Conv2d(depth * 4, 1, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, input, label):
        network = self.conv1_1(input)
        network = leaky_relu(network, 0.2)

        network_branch = self.conv1_2(label)
        network_branch = leaky_relu(network_branch, 0.2)

        network = torch.cat([network, network_branch], 1)

        network = self.conv2(network)
        network = self.conv2_bn(network)
        network = leaky_relu(network, 0.2)

        network = self.conv3(network)
        network = self.conv3_bn(network)
        network = leaky_relu(network, 0.2)

        network = self.conv4(network)
        network = sigmoid(network)

        return network

def normal_init(m, mean, std):
    if isinstance(m, ConvTranspose2d) or isinstance(m, Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    Net_G.eval()
    test_images = Net_G(fixed_z_, fixed_y_label_)
    Net_G.train()

    size_figure_grid = 10
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(10*10):
        i = k // 10
        j = k % 10
        ax[i, j].cla()
        ax[i, j].imshow(test_images[k, 0].cpu().data.numpy(), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# fixed noise & label
temp_z_ = torch.randn(10, 100)
fixed_z_ = temp_z_
fixed_y_ = torch.zeros(10, 1)
for i in range(9):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp = torch.ones(10, 1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)

fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
fixed_y_label_ = torch.zeros(100, 10)
fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
fixed_y_label_ = fixed_y_label_.view(-1, 10, 1, 1)
fixed_z_, fixed_y_label_ = Variable(fixed_z_.cuda() if GPU_NUMS > 1 else fixed_z_, volatile=True), Variable(fixed_y_label_.cuda() if GPU_NUMS > 1 else fixed_y_label_, volatile=True)

# training parameters
BATCH_SIZE = 100
LR = 0.0002
EPOCHS = 20

# data_loader
IMG_SIZE = 32
'''
生成网络
'''
Net_G = Generator(depth=128)
Net_D = Discriminator(depth=128)
Net_G.weight_init(mean=0.0, std=0.02)
Net_D.weight_init(mean=0.0, std=0.02)

Net_G = DataParallel(Net_G)
Net_D = DataParallel(Net_D)
if GPU_NUMS > 1:
    Net_G.cuda()
    Net_D.cuda()
'''
读入数据并进行预处理
'''
transform = Compose([
    Scale(IMG_SIZE),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = torch.utils.data.DataLoader(
    # MNIST('data', train=True, download=True, transform=transform),
    MNISTDataSet('../ganData/mnist.npz', train=True, transform=transform),
    batch_size=BATCH_SIZE, shuffle=True)


'''
开始训练
'''
BCE_loss = BCELoss()
G_optimizer = Adam(Net_G.parameters(), lr=LR, betas=(0.5, 0.999))
D_optimizer = Adam(Net_D.parameters(), lr=LR, betas=(0.5, 0.999))
bar = ProgressBar(EPOCHS, len(train_loader), "D Loss:%.3f; G Loss:%.3f")

# label preprocess
onehot = torch.zeros(10, 10)
onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
fill = torch.zeros([10, 10, IMG_SIZE, IMG_SIZE])

for i in range(10):
    fill[i, i, :, :] = 1

bar = ProgressBar(EPOCHS, len(train_loader), "D Loss:%.3f; G Loss:%.3f")
for epoch in range(EPOCHS):
    # learning rate decay
    if (epoch+1) == 11:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    if (epoch+1) == 16:
        G_optimizer.param_groups[0]['lr'] /= 10
        D_optimizer.param_groups[0]['lr'] /= 10
        print("learning rate change!")

    # y_real_ = torch.ones(BATCH_SIZE)
    # y_fake_ = torch.zeros(BATCH_SIZE)
    # y_real_, y_fake_ = Variable(y_real_.cuda() if GPU_NUMS > 1 else y_real_), Variable(y_fake_.cuda() if GPU_NUMS > 1 else y_fake_)
    label_true = torch.ones(BATCH_SIZE)
    label_false = torch.zeros(BATCH_SIZE)
    label_true_var  = Variable(label_true.cuda() if GPU_NUMS > 1 else label_true)
    label_false_var = Variable(label_false.cuda() if GPU_NUMS > 1 else label_false)
    for img_real, label_real in train_loader:
        Net_D.zero_grad()
        label_real = label_real.type(torch.LongTensor)

        '''
         真值损失
         '''
        label_real = fill[label_real] # 将[BATCH_SIZE]，变为=>[BATCH_SIZE,10,IMAGE_SIZE,IMAGE_SIZE]
        image_real_var = Variable(img_real.cuda() if GPU_NUMS > 1 else img_real)
        label_real_var = Variable(label_real.cuda() if GPU_NUMS > 1 else label_real)
        D_result = Net_D(image_real_var, label_real_var).squeeze()
        D_real_loss = BCE_loss(D_result, label_true_var)



        '''
        假值损失
        '''
        img_fake = torch.randn((BATCH_SIZE, 100)).view(-1, 100, 1, 1)
        label_fake = (torch.rand(BATCH_SIZE, 1) * 10).type(torch.LongTensor).squeeze()
        img_fake_var = Variable(img_fake.cuda() if GPU_NUMS > 1 else img_fake)
        label_fake_G_var = Variable(onehot[label_fake].cuda() if GPU_NUMS > 1 else onehot[label_fake])
        label_fake_D_var = Variable(fill[label_fake].cuda() if GPU_NUMS > 1 else fill[label_fake])
        G_result = Net_G(img_fake_var, label_fake_G_var)
        D_result = Net_D(G_result, label_fake_D_var).squeeze()
        D_fake_loss = BCELoss()(D_result, label_false_var)

        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.backward()
        D_optimizer.step()

        '''
        生成器训练
        '''
        Net_G.zero_grad()
        img_fake = torch.randn((BATCH_SIZE, 100)).view(-1, 100, 1, 1)
        label_fake = (torch.rand(BATCH_SIZE, 1) * 10).type(torch.LongTensor).squeeze() # [BATCH_SIZE]
        img_fake_var = Variable(img_fake.cuda() if GPU_NUMS > 1 else img_fake)
        label_fake_G_var = Variable(onehot[label_fake].cuda() if GPU_NUMS > 1 else onehot[label_fake]) #[BATCH,10,1,1]
        label_fake_D_var = Variable(fill[label_fake].cuda() if GPU_NUMS > 1 else fill[label_fake]) #[BATCH,10,IMAGE_SIZE,IMAGE_SIZE]
        G_result = Net_G(img_fake_var, label_fake_G_var)
        D_result = Net_D(G_result, label_fake_D_var).squeeze()
        G_train_loss= BCELoss()(D_result, label_true_var)
        G_train_loss.backward()
        G_optimizer.step()

        bar.show(D_train_loss.data[0], G_train_loss.data[0])

    fixed_p = 'Fixed_results/' + str(epoch + 1) + '.png'
    show_result((epoch+1), save=True, path=fixed_p)
    torch.save(Net_G.state_dict(),'Fixed_results/netg_%s.pth' % epoch)