# coding=utf-8
import torch
from torch import nn as nn
from torch.utils import data as data
from torch.autograd import Variable
import torchvision
from lib.datareader.pytorch.MNIST import MNISTDataSet
from torch.optim import Adam
from library.keras_callbacks import ProgressBarCallback as bar
from torch.nn import functional as F

torch.manual_seed(1)
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.out(x)

# train_data = torchvision.datasets.MNIST(
#     root='../data/MNIST',
#            train=True,  # this is training data
#                  transform=torchvision.transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
#                            # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
#                            download=False,          # 没下载就下载, 下载了就不用再下了
# )

# print(train_data.train_data)
train_data = MNISTDataSet(train=True, transform=torchvision.transforms.ToTensor())
test_data  = MNISTDataSet(train=False)
train_loader = data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE,
                               shuffle=True)
# test_x = Variable(torch.unsqueeze(test_data.test_data, dim=1), volatile=True).type(torch.FloatTensor)[:2000]/255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
# test_y = test_data.test_labels[:2000]

cnn = CNN()
optimizer = Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
proBar = bar.ProgressBarGAN(EPOCH, len(train_loader), "train loss: %.3f | test accuracy: %.3f")
for epoch in range(EPOCH):
    for step, (x,y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y.type(torch.LongTensor))
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction = torch.max(F.softmax(output), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = b_y.data.numpy()
        accuracy = sum(pred_y == target_y) / len(target_y)

        proBar.show(loss.data[0], accuracy)

test_x = Variable(torch.unsqueeze(torch.FloatTensor(test_data.test_data), dim=1), volatile=True).type(torch.FloatTensor)
test_y = Variable(torch.LongTensor(test_data.test_labels), volatile=True).type(torch.LongTensor)
test_output = cnn(test_x)
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
target_y = test_y.data.numpy()
accuracy = sum(pred_y == target_y) / len(target_y)
print("test accuracy is %.3f" % accuracy)