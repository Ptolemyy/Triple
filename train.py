import torch
from torch import nn
import numpy as np
from torch.nn import Sequential,Conv2d,Linear,MaxPool2d,Flatten,ReLU,AvgPool2d,Sigmoid
import Triple as gm

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

class ResNet(nn.Module):#待修改
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

class Node:
    def __init__(self, board, pool, point = 0, num = 0, N=0, ar=0):
        super(Node,self).__init__()
        self.board = board
        self.pool = pool
        self.point = point
        self.ar = ar
        self.num = num
        self.V = []

        gm.set_board(self.board)
        possible_num = gm.possible_num()
        self.num_pool = np.array([possible_num[x % len(possible_num)] for x in pool])
        gm.set_pool = self.num_pool
        
        input1 = np.pad(self.board,pad_width=1,mode="constant",constant_values=0)
        input2 = np.pad(self.num_pool,((0,2)),mode="constant",constant_values=0)
        input2 = np.reshape(input2,(2,2))
        input2 = np.pad(input2,((0,4),(0,4)),mode="constant",constant_values=0)
        input = torch.tensor(input1+input2,dtype=torch.float32)
        input = torch.reshape(input,(1,6,6))
        resnet = ResNet()
        self.output = resnet(input)
        self.P = self.output[:-1:]
        self.V0 = self.output[16]

    def calc(self):
        self.N_b = 0 #下一层节点访问次数总和
        self.V = 1/np.array(self.V)
        self.Q = self.V/max(self.V)

    def make_leafs_prop(self):
        board_1 = np.reshape(self.board,16)
        raw_sliced = 1-np.array([min(x,1) for x in board_1])
        board_list = []
        for i,x in enumerate(raw_sliced):
            board_ = np.copy(board_1)
            board_[i] = x * self.num_pool[0]
            board_ = np.reshape(board_,(4,4))
            board_list.append(board_)
        return board_list

def make_leafs(Tree, pool, father, point):
    prop = father.make_leafs_prop()
    ar = father.ar
    init_pool = pool[ar:ar+2]
    for i,x in enumerate(prop):
        Tree.append(Node(board = x, pool = init_pool, point = point, num = i, ar = ar+1))
    return Tree

pool = np.random.randint(10,99,3000)
tree = []
init_board = np.array([[0,0,0,0],
                       [0,0,81,0],
                       [0,0,0,0],
                       [0,0,0,0]])
init_pool = pool[:2]
tree.append(Node(board = init_board , pool = init_pool, point=-1))

for i, x in enumerate(tree):
    if x.point == -1 and x.V == []:
        tree = make_leafs(tree, pool, x, i)