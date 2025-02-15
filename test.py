import torch
from torch import nn
import numpy as np
from torch.nn import Sequential,Conv2d,Linear,Flatten,ReLU,Sigmoid,BatchNorm1d,MSELoss,CrossEntropyLoss,Softmax
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from Triple import Triple
import time
import tqdm
import json, os, argparse

class BasicNet(nn.Module):
    def __init__(self, channel):
        super(BasicNet, self).__init__()
        self.block1 = Sequential(
            Conv2d(channel, channel, 3, stride=1, padding=1),
            BatchNorm1d(4),
            ReLU(),
            Conv2d(channel, channel, 3, stride=1, padding=1),
            BatchNorm1d(4)
        )
        self.relu = ReLU()

    def forward(self, x):
        identity = x
        x = self.block1(x)
        x += identity
        x = self.relu(x)
        return x

class policy_head(nn.Module):
    def __init__(self, channel):
        super(policy_head, self).__init__()
        self.net = Sequential(
            Conv2d(channel, 2, 1, stride=1),
            BatchNorm1d(4),
            ReLU(),
            Flatten(0),
            Linear(2 * 16, 16),
            Softmax(0)
        )
    def forward(self, x):
        x = self.net(x)
        return x

class value_head(nn.Module):
    def __init__(self, channel):
        super(value_head, self).__init__()
        self.net = Sequential(
            Conv2d(channel, 1, 1, stride=1),
            BatchNorm1d(4),
            ReLU(),
            Flatten(0),
            Linear(16, 256),
            ReLU(),
            Linear(256, 1),
            Sigmoid()
        )
    def forward(self, x):
        x = self.net(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.Convolutional = Sequential(
            Conv2d(8, 256, 3, stride=1, padding=0),
            BatchNorm1d(4),
            ReLU()
        )
        self.net = self.residual_block()
        self.policy = policy_head(256)
        self.value = value_head(256)
    def residual_block(self):
        block = []
        length = 20
        for _ in range(0, length - 1):
            block.append(BasicNet(256))
        return Sequential(*block)

    def forward(self, x):
        x = self.Convolutional(x)
        x = self.net(x)
        p = self.policy(x)
        v = self.value(x)
        output = torch.cat((p, v), 0)
        return output
    
input = torch.randn(8, 6, 6)
resnet = ResNet()
print(resnet(input))