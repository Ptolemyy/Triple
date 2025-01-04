import torch
from torch import nn
import numpy as np
from torch.nn import Sequential,Conv2d,Linear,MaxPool2d,Flatten,ReLU,AvgPool2d,Sigmoid,BatchNorm1d
import Triple as gm
import time

class BasicNet(nn.Module):
    def __init__(self, channel):
        super(BasicNet, self).__init__()
        self.block1 = Sequential(
            Conv2d(channel, channel, 3, stride=1, padding=1),
            BatchNorm1d(6),
            ReLU(),
            Conv2d(channel, channel, 3, stride=1, padding=1),
            BatchNorm1d(6)
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
            BatchNorm1d(6),
            ReLU(),
            Flatten(0),
            Linear(2 * 36, 16),
        )
    def forward(self, x):
        x = self.net(x)
        return x

class value_head(nn.Module):
    def __init__(self, channel):
        super(value_head, self).__init__()
        self.net = Sequential(
            Conv2d(channel, 1, 1, stride=1),
            BatchNorm1d(6),
            ReLU(),
            Flatten(0),
            Linear(36, 256),
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
            Conv2d(8, 256, 3, stride=1, padding=1),
            BatchNorm1d(6),
            ReLU()
        )
        self.net = self.residual_block()
        self.policy = policy_head(256)
        self.value = value_head(256)
    def residual_block(self):
        block = []
        len = 20
        for _ in range(0, len - 1):
            block.append(BasicNet(256))
        return Sequential(*block)

    def forward(self, x):
        x = self.Convolutional(x)
        x = self.net(x)
        p = self.policy(x)
        v = self.value(x)
        return p, v

class Node:
    def __init__(self,  pool, board, feather_planes,resnet,
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
        feather_planes = torch.tensor(feather_planes)

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
            self.input = input.to(torch.float16)
            input = input.cuda()
            feather_planes = feather_planes.cuda()
            r_input = torch.cat((input, feather_planes),0)
            r_input = r_input.to(torch.float16)

            with torch.no_grad():

                self.output = resnet(r_input)
                self.P, self.V0 = self.output
                self.P = self.P.cpu()
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
    def __init__(self,resnet):
        super(Tree, self).__init__()
        self.tree = []
        self.maximum_visit_count = 3
        self.c_puct = 1
        
        self.pool = np.random.randint(10,99,3000)
        init_board = np.array([[0,0,0,0],
                        [0,0,0,0],
                        [0,81,0,0],
                        [0,0,0,0]],dtype=np.float16)
        init_pool = self.pool[:2]
        self.tree.append(Node(board = init_board,
                            pool = init_pool,
                            point=-1,
                            feather_planes=np.full((7,6,6),np.zeros((6,6))),
                            resnet = resnet
                            ))

    def backup(self, arr = 0):
        #print(self.tree[-1].board)
        while True:
            x = self.tree[arr]
            if (x.point == -1 or x.N == self.maximum_visit_count) and np.any(x.V==-1):
                self.make_leafs(x, arr)
                print("expanded",len(self.tree))
            x.V, x.N0 = search_nodes(self.tree, arr)
            #print(x.point)
            #print(x.N0)
            arr0 = self.tree[arr].backup_calc(self.c_puct)

            arr = find_leaf(self.tree, arr, arr0)
            self.tree[arr].N += 1
            if self.tree[arr].N < self.maximum_visit_count:
                break

    def make_leafs(self, father, point):
        prop = father.make_leafs_prop()
        ar = father.ar
        init_pool = self.pool[ar:ar+2]
        feather_planes = self.feather_planes(point)
        for i,x in enumerate(prop):
            gp = i if ar == -1 else father.gp
            self.tree.append(Node(placement = x[:2],
                            pool = init_pool,
                            point = point,
                            num = i,
                            ar = ar+1,
                            board = x[2],
                            gp = gp,
                            feather_planes = feather_planes,
                            resnet = resnet))

    def feather_planes(self, f):
        x = torch.tensor([])
        for _ in range(7):
            Xi = self.tree[f].input
            x = torch.cat((x, Xi if f != -1 else torch.zeros((6,6))), 0)
            f = max(self.tree[f].point, 0)
        return x

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

resnet = ResNet()
resnet = resnet.cuda()
resnet = resnet.half()
tree = Tree(resnet)

temp = time.time()
while True:
    tree.backup()
    if len(tree.tree) > 2000:
        pi = tree.tree[0].N0/max(tree.tree[0].N0)
        print(pi)
        print(time.time()-temp)
        break