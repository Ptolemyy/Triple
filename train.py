import torch
from torch import nn
import numpy as np
from torch.nn import Sequential,Conv2d,Linear,Flatten,ReLU,Sigmoid,BatchNorm1d,L1Loss
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
import Triple as gm
import time
import tqdm
import json, os

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
        return torch.cat([p, v], 0)

class Node:
    def __init__(self,  pool, board, feather_planes, resnet,
                point = 0, num = 0, ar = 0, placement = np.full(2,-1)):
        super(Node,self).__init__()
        self.placement = placement
        self.pool = pool
        self.point = point
        self.ar = ar
        self.N = 0
        self.num = num
        self.V = np.full(16,-1)
        self.N0 = np.array([])
        self.board = board
        self.order = time.time()
        self.order += 100000 * np.random.random()
        self.feather_planes = feather_planes

        gm.board = np.copy(self.board)
        gm.board = np.reshape(gm.board, 16)
        if placement[0] >= 0:
            gm.place(placement[0], placement[1])
        if placement[0] == -2:
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
            self.r_input = torch.cat((input, feather_planes),0)
            self.r_input = self.r_input.to(torch.float16)
            with torch.no_grad():

                self.output = resnet(self.r_input)
                self.P = self.output[:-1]
                self.V0 = self.output[-1]
                self.P = self.P.cpu()
                self.V0 = self.V0.cpu()
        else:
            self.V0 = 0
            self.P = torch.zeros(16)
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
            placement = -2, -2, self.board
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
                        [0,0,0,0],
                        [0,0,0,0]],dtype=np.float16)
        init_pool = self.pool[:2]
        feather_planes = np.full((7,6,6),np.zeros((6,6)))
        feather_planes = torch.tensor(feather_planes)

        self.tree.append(Node(board = init_board,
                            pool = init_pool,
                            point=-1,
                            feather_planes = feather_planes,
                            resnet = resnet
                            ))

    def backup(self):
        #print(self.tree[-1].board)
        x = self.tree[0]
        j = x.order
        while True:
            if ((x.point == -1 or x.N == self.maximum_visit_count)
                    and np.any(x.V == -1) and np.any(x.board != -1)):
                self.make_leafs(x, j)
            x.V, x.N0 = self.search_nodes(j)
            #print(x.N0,x.N)
            #print(x.point)
            #print(x.N0)
            if np.any(x.board != -1):
                arr0 = x.backup_calc(self.c_puct)
                j0 = self.find_leaf(j, arr0)
                x = self.tree[j0]
            j = x.order
            x.N += 1
            if (x.N < self.maximum_visit_count
                    or np.any(x.board == -1)):
                break

    def make_leafs(self, father, point):
        prop = father.make_leafs_prop()
        ar = father.ar
        init_pool = self.pool[ar:ar+2]
        feather_planes = father.r_input[:-1]
        for i,x in enumerate(prop):
            self.tree.append(Node(placement = x[:2],
                            pool = init_pool,
                            point = point,
                            num = i,
                            ar = ar + 1,
                            board = x[2],
                            feather_planes = feather_planes,
                            resnet = resnet))

    def restart(self, select):
        temp_tree = []
        list0 = [self.tree[select+1].order]
        while True:
            for i,x in enumerate(list0):
                if x != 0:
                    new_list = [y.order for y in self.tree if y.point == x]
                    list0[i] = 0
                    list0 += new_list
                    node0 = [y for y in self.tree if y.order == x]
                    temp_tree.append(node0[0])
            if not any(list0):
                break
        self.tree = temp_tree

    def search_nodes(self, point):
        V = np.zeros(16)
        N = np.zeros(16)
        for i in self.tree:
            if i.point == point and np.any(i.board!=-1):        #查找所有合法叶节点
                V[i.num] = i.V0
                N[i.num] = i.N
        return V, N

    def find_leaf(self, point, num):
        for i, x in enumerate(self.tree):
            if x.point == point and x.num == num:
                return i


def single_move(saved, title):
    total_visit_count = 700

    p_bar = tqdm.tqdm(total=total_visit_count - saved, desc=title)

    while True:
        N0 = tree.tree[0].N0
        tree.backup()
        p_bar.update(1)
        p_bar.refresh()

        if  sum(N0) > total_visit_count:
            pi = N0 / max(N0)
            select_move = np.argmax(pi)
            tree.restart(select_move)
            p_bar.close()
            return sum(tree.tree[0].N0)

def self_play(title):
    saved_move = 0
    self_play_dict = []

    while True:
        tree_root = tree.tree[0]
        bd = tree_root.board
        if not np.any(bd==0):
            break
        saved_move = single_move(saved_move, title)
        pi = tree_root.N0 / max(tree_root.N0)
        dict0 = {"board": tree_root.r_input.tolist(), "Pi": pi.tolist(), "P": 0}
        self_play_dict.append(dict0)

    for i, x in enumerate(self_play_dict):
        x["P"] = len(self_play_dict) - i

    name = time.asctime().replace(" ", "_")
    name = name.replace(":", "_")
    with open("data/" + name + ".json", 'w') as f:
        f.write(json.dumps(self_play_dict, indent=4))

    return len(self_play_dict)

def get_data():
    data = []
    with open("data/log.json","r") as f:
        log = json.loads(f.read())
    entries = os.listdir("data")
    for i in entries:
        if (not i in log) and i != "log.json":
            log.append(i)
            with open("data/" + i, "r") as f:
                data += json.loads(f.read())
    with open("data/log.json", 'w') as f:
        f.write(json.dumps(log))

    return data

class TripleDataset(Dataset):
    def __init__(self):
        self.data = get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["board"]
        img = torch.tensor(img,dtype=torch.float16)
        target_pi = self.data[idx]["Pi"]
        target_pi = torch.tensor(target_pi,dtype=torch.float16)
        target_p = 1/self.data[idx]["P"]
        target_p = torch.tensor([target_p],dtype=torch.float16)
        outputs = torch.cat((target_pi,target_p),dim=0)
        return img, outputs

def train():
    dataloader = DataLoader(batch_size=1, shuffle=True, dataset=TripleDataset())
    loss = L1Loss()
    loss.cuda()
    optim = SGD(resnet.parameters(), lr=0.005)

    for data in dataloader:
        img, target = data
        img = img[0].cuda()
        target = target[0].cuda()
        outputs = resnet(img)

        result_loss = loss(outputs, target)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        print(result_loss.item())

if __name__ == "__main__":
    model_name = "demo"
    model_path = "model/"
    stat_path = "stat/"
    self_play_count = 5

    with open(stat_path + model_name + ".json", 'r') as f:
        mean_move_list = json.loads(f.read())

    while True:
        resnet = ResNet()
        if model_name + ".pt" in os.listdir(model_path):
            print(model_name+ " has been loaded")
            resnet.load_state_dict(torch.load(model_path + model_name + ".pt"))
        resnet = resnet.cuda()
        resnet = resnet.half()

        move_sum = 0
        for i in range(self_play_count):
            tree = Tree(resnet)
            move_sum += self_play("self_play_count:" + str(i))
        mean_move_list.append(move_sum / self_play_count)

        with open(stat_path + model_name + ".json", 'w') as f:
            f.write(json.dumps(mean_move_list))

        train()
        torch.save(resnet.state_dict(),model_path + model_name + ".pt")
