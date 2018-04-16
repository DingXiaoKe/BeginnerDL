# coding=utf-8
import numpy as np
import torch
import os

from torch.nn import Conv2d, ConvTranspose2d, Linear, BatchNorm1d, BatchNorm2d, ReLU, Sigmoid,Module, LeakyReLU,\
    DataParallel, NLLLoss, MSELoss, BCELoss, Softmax, LogSoftmax
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
plt.switch_backend('agg')

from lib.utils.progressbar.ProgressBar import ProgressBar
from lib.datareader.common import read_mnist
GPU_NUMS = 2
BATCH_SIZE = 128
EPOCHS = 30
c1_len = 10 # Multinomial
c2_len = 2 # Gaussian
c3_len = 0 # Bernoulli
z_len = 64 # Noise vector length
embedding_len = 128

'''
定义层
'''
class Conv2d(Conv2d):
    def reset_parameters(self):
        stdv = np.sqrt(6 / ((self.in_channels  + self.out_channels) * np.prod(self.kernel_size)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

class ConvTranspose2d(ConvTranspose2d):
    def reset_parameters(self):
        stdv = np.sqrt(6 / ((self.in_channels  + self.out_channels) * np.prod(self.kernel_size)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

class Linear(Linear):
    def reset_parameters(self):
        stdv = np.sqrt(6 / (self.in_features + self.out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
'''
定义生成器
'''
class Generator(Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = Linear(z_len + c1_len + c2_len + c3_len, 1024)
        self.fc2 = Linear(1024, 7 * 7 * 128)

        self.convt1 = ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1)
        self.convt2 = ConvTranspose2d(64, 1, kernel_size = 4, stride = 2, padding = 1)

        self.bn1 = BatchNorm1d(1024)
        self.bn2 = BatchNorm1d(7 * 7 * 128)
        self.bn3 = BatchNorm2d(64)

    def forward(self, x):
        x = ReLU()(self.bn1(self.fc1(x)))
        x = ReLU()(self.bn2(self.fc2(x))).view(-1, 128, 7, 7)

        x = ReLU()(self.bn3(self.convt1(x)))
        x = self.convt2(x)

        return Sigmoid()(x)

'''
定义判别器
'''
class Discriminator(Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = Conv2d(1, 64, kernel_size = 4, stride = 2, padding = 1) # 28 x 28 -> 14 x 14
        self.conv2 = Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1) # 14 x 14 -> 7 x 7

        self.fc1 = Linear(128 * 7 ** 2, 1024)
        self.fc2 = Linear(1024, 1)
        self.fc1_q = Linear(1024, embedding_len)

        self.bn1 = BatchNorm2d(128)
        self.bn2 = BatchNorm1d(1024)
        self.bn_q1 = BatchNorm1d(embedding_len)

    def forward(self, x):
        x = LeakyReLU()(self.conv1(x))
        x = LeakyReLU()(self.bn1(self.conv2(x))).view(-1, 7 ** 2 * 128)

        x = LeakyReLU()(self.bn2(self.fc1(x)))
        return self.fc2(x), LeakyReLU()(self.bn_q1(self.fc1_q(x)))

'''
生成网络
'''
Net_G = Generator()
Net_D = Discriminator()
Net_G = DataParallel(Net_G)
Net_D = DataParallel(Net_D)
Q_cat = Linear(embedding_len, c1_len)

if GPU_NUMS > 1:
    Net_G = Generator().cuda()
    Net_D = Discriminator().cuda()
    Q_cat = Q_cat.cuda()

qcat_optim = Adam(Q_cat.parameters(), lr = 2e-4)
if c2_len:
    Q_con = Linear(embedding_len, c2_len).cuda() if GPU_NUMS > 1 else Linear(embedding_len, c2_len)
    qcon_optim = Adam(Q_con.parameters(), lr = 2e-4)
if c3_len:
    Q_bin = Linear(embedding_len, c3_len).cuda() if GPU_NUMS > 1 else Linear(embedding_len, c3_len)
    qbin_optim = Adam(Q_bin.parameters(), lr = 2e-4)

g_optim = Adam(Net_G.parameters(), lr = 1e-3)
d_optim = Adam(Net_D.parameters(), lr = 2e-4)

nll = NLLLoss().cuda() if GPU_NUMS > 1 else NLLLoss()
mse = MSELoss().cuda() if GPU_NUMS > 1 else MSELoss()
bce = BCELoss().cuda() if GPU_NUMS > 1 else BCELoss()

'''
读取数据
'''
(X_train, Y_train), (X_test, Y_test) = read_mnist("/data/mnist.npz")
x_train = np.expand_dims(X_train, 1)
y_train = np.zeros((Y_train.shape[0], 10), dtype = np.uint8)
y_train[np.arange(Y_train.shape[0]), Y_train] = 1

x_test = np.expand_dims(X_test, 1)
y_test = np.zeros((X_test.shape[0], 10), dtype = np.uint8)
y_test[np.arange(Y_test.shape[0]), Y_test] = 1

x_train = x_train.astype(np.uint8)
x_test = x_test.astype(np.uint8)
y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)

supervision = 100 # Number of samples to supervise with

# Prep the data by turning them into tensors and putting them into a PyTorch dataloader
shuffle_train = np.random.permutation(y_train.shape[0])
x_train_th = torch.from_numpy(x_train[shuffle_train])
y_train_th = torch.from_numpy(y_train[shuffle_train]).float()

x_test_th = torch.from_numpy(x_test)
y_test_th = torch.from_numpy(y_test)

# OK, we're going to be hacking this out. We'll multiply by the sum of the labels
# So to make this semisupervised, we set the labels we don't want to 0
y_train_th[int(supervision):] = 0

train_tensors = TensorDataset(x_train_th, y_train_th)
test_tensors = TensorDataset(x_test_th, y_test_th)
train_loader = DataLoader(train_tensors, batch_size = BATCH_SIZE, shuffle = True, num_workers = 6, pin_memory = True)
test_loader = DataLoader(test_tensors, batch_size = BATCH_SIZE, shuffle = True, num_workers = 6, pin_memory = True)

'''
开始训练
'''
def get_z(length, sequential = False):
    weights = torch.Tensor([0.1] * 10)

    z = {}
    if z_len:
        z['z'] = Variable(torch.randn(length, z_len)).cuda() if GPU_NUMS > 1 else Variable(torch.randn(length, z_len))

    if c1_len:
        if sequential:
            cat_noise = Variable(torch.arange(0, c1_len).repeat(length // c1_len).long()).cuda() if GPU_NUMS > 1 else Variable(torch.arange(0, c1_len).repeat(length // c1_len).long())
        else:
            cat_noise = Variable(torch.multinomial(weights, num_samples = length, replacement = True)).cuda().view(-1) if GPU_NUMS > 1 else Variable(torch.multinomial(weights, num_samples = length, replacement = True)).view(-1)

        onehot_noise = Variable(torch.zeros(length, c1_len)).cuda() if GPU_NUMS > 1 else Variable(torch.zeros(length, c1_len))
        onehot_noise.data.scatter_(1, cat_noise.data.view(-1, 1), 1)
        z['cat'] = onehot_noise

    if c2_len:
        z['con'] = Variable(torch.rand(length, c2_len)).cuda() * 2 - 1 if GPU_NUMS > 1 else Variable(torch.rand(length, c2_len)) * 2 - 1

    if c3_len:
        z['bin'] = Variable(torch.bernoulli(0.5 * torch.ones(length, c3_len))).cuda().float() if GPU_NUMS > 1 else Variable(torch.bernoulli(0.5 * torch.ones(length, c3_len))).float()

    return z


def run_dis(x):
    out = []
    out_dis, hid = Net_D(x)
    out += [out_dis]
    if c1_len:
        out += [Softmax()(Q_cat(hid))]
    if c2_len:
        out += [Q_con(hid)]
    if c3_len:
        out += [Sigmoid()(Q_bin(hid))]

    return out


def save(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(Net_G.state_dict(), directory + 'gen.torch')
    torch.save(Net_D.state_dict(), directory + 'dis.torch')
    if c1_len:
        torch.save(Q_cat.state_dict(), directory + 'qcat.torch')
    if c2_len:
        torch.save(Q_con.state_dict(), directory + 'qcon.torch')
    if c3_len:
        torch.save(Q_bin.state_dict(), directory + 'qbin.torch')


def load( directory):
    Net_G.load_state_dict(torch.load(directory + 'gen.torch'))
    Net_D.load_state_dict(torch.load(directory + 'dis.torch'))
    if c1_len:
        Q_cat.load_state_dict(torch.load(directory + 'qcat.torch'))
    if c2_len:
        Q_con.load_state_dict(torch.load(directory + 'qcon.torch'))
    if c3_len:
        Q_bin.load_state_dict(torch.load(directory + 'qbin.torch'))

def sample_images(epoch):
    r, c = 10, 10

    fig, axs = plt.subplots(r, c)
    for i in range(c):
        z_dict = get_z(c1_len * 10, sequential = True)
        out_gen = Net_G(torch.cat([z_dict[k] for k in z_dict.keys()], dim = 1))

        for j in range(r):
            idx = i * 10 + j + 1
            axs[j,i].imshow(np.round(out_gen[idx - 1, 0].cpu().data.numpy() * 255), cmap = 'gray')
            axs[j,i].axis('off')
    fig.savefig("output/mnist_%02d.png" % epoch)
    plt.close()

bar = ProgressBar(EPOCHS, len(train_loader), "D Loss:%.3f;G Loss:%.3f;Q Loss:%.3f")
for epoch in range(EPOCHS):
    for i, (data, targets) in enumerate(train_loader):
        ones = Variable(torch.ones(data.size()[0], 1)).cuda() if GPU_NUMS > 1 else Variable(torch.ones(data.size()[0], 1))
        zeros = Variable(torch.zeros(data.size()[0], 1)).cuda() if GPU_NUMS > 1 else Variable(torch.zeros(data.size()[0], 1))

        z_dict = get_z(data.size()[0])
        z = torch.cat([z_dict[k] for k in z_dict.keys()], dim = 1)

        data = Variable(data.float().cuda(async = True) if GPU_NUMS > 1 else data.float()) / 255
        targets = Variable(targets.float().cuda(async = True) if GPU_NUMS > 1 else targets.float())

        # Forward pass on real MNIST
        out_dis, hid = Net_D(data)
        c1 = LogSoftmax()(Q_cat(hid))
        loss_dis = mse(out_dis, ones) - torch.sum(targets * c1) / (torch.sum(targets) + 1e-3) # Loss for real MNIST

        # Forward pass on generated MNIST
        out_gen = Net_G(z)
        out_dis, hid = Net_D(out_gen)

        # Loss for generated MNIST
        loss_dis = loss_dis + mse(out_dis, zeros)
        loss_dis = loss_dis

        # Zero gradient buffers for gen and Q_cat and backward pass
        Net_D.zero_grad()
        Q_cat.zero_grad()
        loss_dis.backward(retain_graph = True) # We need PyTorch to retain the graph buffers so we can run backward again later
        d_optim.step() # Apply the discriminator's update now since we have to delete its gradients later

        # And backward pass and loss for generator and update
        Net_G.zero_grad()
        loss_gen = mse(out_dis, ones)
        loss_gen.backward(retain_graph = True)
        Net_D.zero_grad() # Don't want the gradients of the generator's objective in the discriminator

        # Forward pass and loss for latent codes
        loss_q = 0

        c1 = LogSoftmax()(Q_cat(hid))
        loss_q += nll(c1, torch.max(z_dict['cat'], dim = 1)[1])

        if c2_len:
            c2 = Q_con(hid)
            loss_q += 0.5 * mse(c2, z_dict['con']) # Multiply by 0.5 as we treat targets as Gaussian (and there's a coefficient of 0.5 when we take logs)
            Q_con.zero_grad() # Zero gradient buffers before the backward pass
        if c3_len:
            c3 = Sigmoid()(Q_bin(hid))
            loss_q += bce(c3, z_dict['bin'])
            Q_bin.zero_grad() # Zero gradient buffers before the backward pass

        # Backward pass for latent code objective
        loss_q.backward()

        # Do the updates for everything
        d_optim.step()
        g_optim.step()
        qcat_optim.step()

        if c2_len:
            qcon_optim.step()
        if c3_len:
            qbin_optim.step()

        bar.show(loss_dis.cpu().data.numpy()[0], loss_gen.cpu().data.numpy()[0], loss_q.cpu().data.numpy()[0])

    sample_images(epoch)

torch.save(Net_G.state_dict(),"output/Net_G_cifar_%d.pth" % epoch)

out_test = run_dis(Variable(x_test_th).cuda().float() / 255 if GPU_NUMS > 1 else Variable(x_test_th).cuda().float() / 255)[1]
out_test = np.argmax(out_test.data.cpu().numpy(), axis = 1)
print(np.mean(out_test == np.argmax(y_test, axis = 1)))