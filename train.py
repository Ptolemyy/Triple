import torch
from torch import nn
import numpy as np
from torch.nn import Sequential,Conv2d,Linear,MaxPool2d,Flatten,ReLU,AvgPool2d,Sigmoid

input1 = np.array([[3,0,0,0],
                   [0,3,3,0],
                   [0,0,0,0],
                   [0,0,0,0]])
input2 = np.array([1,1])

input1 = np.pad(input1,pad_width=1,mode="constant",constant_values=0)
input2 = np.pad(input2,((0,2)),mode="constant",constant_values=0)
input2 = np.reshape(input2,(2,2))
input2 = np.pad(input2,((0,4),(0,4)),mode="constant",constant_values=0)
input = torch.tensor(input1+input2,dtype=torch.float32)
input = torch.reshape(input,(1,6,6))

class BasicNet(nn.Module):
    def __init__(self,channel):
        super(BasicNet, self).__init__()
        self.block1 = Sequential(
            Conv2d(channel,channel,3,stride=1,padding=1),
            ReLU(),
            Conv2d(channel,channel,3,stride=1,padding=1)
        )
        self.relu = ReLU()
    def forward(self,x):
        identity = x
        x = self.block1(x)
        x += identity
        x = self.relu(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Bottleneck,self).__init__()
        self.block2 = Sequential(
            Conv2d(in_channel,out_channel,3,2,1),
            ReLU(),
            Conv2d(out_channel,out_channel,3,1,1)
        )
        self.connect = Conv2d(in_channel,out_channel,1,stride=2)
        self.relu = ReLU()
    def forward(self,x):
        identity = self.connect(x)
        x = self.block2(x)
        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.initialize = Sequential(
            Conv2d(1,256,3,stride = 1,padding = 1),
            MaxPool2d(3,stride = 1,padding = 1),
        )
        self.net = Sequential(
            BasicNet(256),BasicNet(256),BasicNet(256),BasicNet(256),BasicNet(256),
            Bottleneck(256,512),
            BasicNet(512),BasicNet(512),BasicNet(512),BasicNet(512),BasicNet(512),
            Bottleneck(512,1024)
        )
        self.ending = Sequential(
            AvgPool2d(2),
            Flatten(0),
            Linear(1024,17),
            Sigmoid()
        )
    def forward(self,x):
        x = self.initialize(x)
        x = self.net(x)
        x = self.ending(x)
        return x
net = ResNet()
output = net(input)
print(output)