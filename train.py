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

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.initialize = Sequential(
            Conv2d(1,256,3,stride = 1,padding = 1),
            MaxPool2d(3,stride = 1,padding = 1),
        )
        self.net = self.residual_block()
        self.ending = Sequential(
            AvgPool2d(2),
            Flatten(0),
            Linear(256,17),
            Sigmoid()
        )
    def residual_block(self):
        block = []
        len = 20
        for _ in range(0,len-1):
            block.append(BasicNet(256))
        block.append(Bottleneck(256,256))
        block = block * 2

        return Sequential(*block)
    
    def forward(self,x):
        x = self.initialize(x)
        x = self.net(x)
        x = self.ending(x)
        return x

class Node:
    def __init__(self,  pool, board,
                  gp = -1, point = 0, num = 0, ar=0, placement = np.full(2,-1)):
        super(Node,self).__init__()
        self.placement = placement
        self.pool = pool
        self.point = point
        self.ar = ar
        self.N = 0
        self.num = num
        self.V = np.full(16,-1)
        self.N0 = []
        self.board = board
        self.gp = gp
        
        gm.board = np.copy(self.board)
        gm.board = np.reshape(gm.board, 16)
        if placement[0] >= 0:
            gm.place(placement[0], placement[1])
        if np.any(placement == -2):
            gm.board = np.full(16, -1)
        self.board = np.copy(gm.board)
        self.board = np.reshape(self.board, (4, 4))

        possible_num = gm.possible_num()
        self.num_pool = np.array([possible_num[x % len(possible_num)] for x in pool])
        gm.set_pool = self.num_pool

        if np.any(self.board!=-1):
            input1 = np.pad(self.board,pad_width=1,mode="constant",constant_values=0)
            input2 = np.pad(self.num_pool,((0,2)),mode="constant",constant_values=0)
            input2 = np.reshape(input2,(2,2))
            input2 = np.pad(input2,((0,4),(0,4)),mode="constant",constant_values=0)
            input = torch.tensor(input1 + input2,dtype=torch.float32)
            input = torch.reshape(input,(1,6,6))
            input = torch.tensor(input,dtype=torch.float16)
            input = input.cuda()

            with torch.no_grad():
                resnet = ResNet()
                resnet = resnet.cuda()
                resnet = resnet.half()
                self.output = resnet(input)
                self.P = self.output[:-1:].detach()
                self.P = self.P.cpu()
                self.V0 = self.output[16]
                self.V0 = self.V0.cpu()
        else:
            self.V0 = 0
    def backup_calc(self, c):
        self.N_s = np.sum(self.N0)
        self.V = np.array(self.V)
        self.V = np.where(self.V == 0, np.inf, self.V)
        self.V = 1/self.V
        self.Q = self.V/max(self.V)
        self.puct = (self.N_s ** 1/2) / (np.ones(16) + self.N0)
        self.U = c * self.P * self.puct
        return torch.argmax(self.U + self.Q)

    def make_leafs_prop(self):
        board_1 = np.reshape(self.board,16)
        raw_sliced = 1 - np.array([min(x,1) for x in board_1])
        placement_list = []
        self.P = self.P * raw_sliced                        #不合法节点
        for i,x in enumerate(raw_sliced):
            placement = -1, -1, self.board
            if x != 0:                                      #合法节点
                placement = i, self.num_pool[0], self.board
            placement_list.append(placement)
        return placement_list

class Tree:
    def __init__(self, init_board):
        super(Tree, self).__init__()
        self.tree = []
        self.maxium_visit_count = 3
        self.c_puct = 1
        
        self.pool = np.random.randint(10,99,3000)
        init_board = np.array([[0,0,0,0],
                        [0,0,0,0],
                        [0,81,0,0],
                        [0,0,0,0]],dtype=np.float16)
        init_pool = self.pool[:2]
        self.tree.append(Node(board = init_board,
                            pool = init_pool,
                            point=-1))
    def backup(self, arr = 0):
        #print(self.tree[-1].board)
        while True:
            x = self.tree[arr]
            if (x.point == -1 or x.N == self.maxium_visit_count) and np.any(x.V==-1):
                self.make_leafs(x, arr)
                print("expanded",len(self.tree))
            x.V, x.N0 = search_nodes(self.tree, arr)
            #print(x.point)
            #print(x.N0)
            arr0 = self.tree[arr].backup_calc(self.c_puct)

            arr = find_leaf(self.tree, arr, arr0)
            self.tree[arr].N += 1
            if self.tree[arr].N < self.maxium_visit_count:
                break

    def make_leafs(self, father, point):
        prop = father.make_leafs_prop()
        ar = father.ar
        init_pool = self.pool[ar:ar+2]
        for i,x in enumerate(prop):
            if ar == -1:
                gp = i
            else:
                gp = father.gp
            self.tree.append(Node(placement = x[:2],
                            pool = init_pool,
                            point = point,
                            num = i,
                            ar = ar+1,
                            board = x[2],
                            gp = gp))


def search_nodes(Tree, point):
    V = np.zeros(16)
    N = np.zeros(16)
    for i in Tree:
        if i.point == point and np.any(i.board!=-1):        #查找所有合法叶节点
            V[i.num] = i.V0
            N[i.num] = i.N
    return V, N

def find_leaf(Tree, point, num):
    for i, x in enumerate(Tree):
        if x.point == point and x.num == num:
            return i

tree = Tree()
while True:
    tree.backup()
    if len(tree.tree) > 1500:
        pi = tree.tree[0].N0/max(tree.tree[0].N0)
        print(pi)
        break
