# coding=utf-8
import torch
from torchvision.models import alexnet, vgg16, vgg19, resnet50, densenet121
from lib.datareader.pytorch.cifar import Cifar10DataSet
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
from torch.nn.functional import softmax
import torchvision.datasets.cifar
from lib.utils.progressbar.ProgressBar import ProgressBar

from lib.models.pytorch.cifar import SENet

EPOCH = 20
BATCH_SIZE = 100
LR = 0.001
MODEL = "senet"

MODEL_LIST = [
    {
        "name": "alexnet",
        "model": alexnet,
        "pretrained" : True,
        "transform" : torchvision.transforms.Compose([torchvision.transforms.Resize(256),
                                                      torchvision.transforms.RandomCrop(224),
                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize([0.485, 0.456, -.406],[0.229, 0.224, 0.225])
                                                      ])
    },
    {
        "name": "vgg16",
        "model": vgg16,
        "pretrained" : False,
        "transform" : torchvision.transforms.ToTensor()
    },
    {
        "name": "vgg19",
        "model": vgg19,
        "pretrained" : False,
        "transform" : torchvision.transforms.ToTensor()
    },
    {
        "name" : "resnet50",
        "model" : resnet50,
        "pretrained" : False,
        "transform" : torchvision.transforms.ToTensor()
    },
    {
        "name" : "densenet121",
        "model" : densenet121,
        "pretrained" : False,
        "transform" : torchvision.transforms.ToTensor()
    },
    {
        "name" : "senet",
        "model" : SENet,
        "pretrained" : False,
        "transform" : torchvision.transforms.ToTensor()
    }
]

# 准备数据
json = MODEL_LIST["name" == MODEL]
train_data = Cifar10DataSet(train=True, transform=json["transform"])
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 准备网络
model = json["model"](json["pretrained"])
model = torch.nn.DataParallel(model).cuda()
optimizer = Adam(model.parameters(), lr=LR)
loss_func = CrossEntropyLoss().cuda()

# 训练数据
proBar = ProgressBar(EPOCH, len(train_loader), "loss:%.3f,acc:%.3f")
for epoch in range(EPOCH):
    for step, (x,y) in enumerate(train_loader):
        data = Variable(x)
        label = Variable(torch.squeeze(y, dim=1).type(torch.LongTensor))
        output = model(data)
        loss = loss_func(output, label)
        loss.backward()
        optimizer.step()

        prediction = torch.max(softmax(output), 1)[1]
        pred_label = prediction.data.numpy().squeeze()
        target_y = label.data.numpy()
        accuracy = sum(pred_label == target_y) / len(target_y)

        proBar.show(loss.data[0], accuracy)