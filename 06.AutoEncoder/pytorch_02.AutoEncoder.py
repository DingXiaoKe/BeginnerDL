# coding=utf-8
from torch import nn
from torch.autograd import Variable
from torch.utils import data
import torchvision
from lib.utils.progressbar.ProgressBar import ProgressBar
from lib.datareader.pytorch.MNIST import MNISTDataSet
from torch.optim import Adam

EPOCH = 10
BATCH_SIZE = 64
LR = 0.005
N_TEST_IMG = 5

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
autoencoder = AutoEncoder()
optimizer = Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

train_data = MNISTDataSet(train=True, transform=torchvision.transforms.ToTensor())
train_loader = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
proBar = ProgressBar(EPOCH, len(train_loader), "loss:%.3f")

for epoch in range(EPOCH):
    for step, (x,y) in enumerate(train_loader):
        b_x = Variable(x.view(-1,28*28))
        b_y = Variable(x.view(-1,28,28))
        b_label = Variable(y)

        encoded, decoded = autoencoder(b_x)
        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        proBar.show(loss.data[0])