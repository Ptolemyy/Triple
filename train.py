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

class Node:
    def __init__(self,  pool, board, feather_planes, resnet, visitcount, epsi = 0.25,
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
        gm = Triple()

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
        if np.any(self.board!=-1):
            input1 = np.pad(self.board,pad_width=1,mode="constant",constant_values=0)
            input2 = np.pad(self.num_pool,(0, 2),mode="constant",constant_values=0)
            input2 = np.reshape(input2,(2,2))
            input2 = np.pad(input2,((0,4),(0,4)),mode="constant",constant_values=0)
            input = torch.tensor(input1 + input2,dtype=torch.float16)
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
                if ar >= 30:
                    np.dtype = np.float16
                    alpha = np.full(16, 0.03)
                    eta = np.random.dirichlet(alpha, 1)[0]
                    self.P = (1 - epsi) * self.P + epsi * eta
                    self.P = torch.tensor(self.P)
                self.V0 = self.V0.cpu()
                self.V0 = 1 / self.V0
        else:
            self.V0 = 0
            self.P = torch.zeros(16)
        self.V0s = np.full(visitcount, self.V0)
    def backup_calc(self, c, visit_count):
        self.N_s = np.sum(self.N0)
        self.Q1 = self.V / (self.N_s + visit_count)
        self.Q = self.Q1 / max(self.Q1)
        self.puct = (self.N_s ** 1/2) / (np.ones(16) + self.N0)
        self.U = c * self.P * self.puct
        #print(self.U, self.Q)
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
    def __init__(self,resnet, maximum_visit_count, c_puct, epsi):
        super(Tree, self).__init__()
        self.tree = []
        self.resnet = resnet
        self.maximum_visit_count = maximum_visit_count
        self.c_puct = c_puct
        self.epsi = epsi

        np.random.seed(int(time.time()))
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
                            resnet = self.resnet,
                            epsi = self.epsi,
                            visitcount = self.maximum_visit_count
                            ))

    def expand_and_evaluate(self):
        #print(self.tree[-1].board)
        x = self.tree[0]
        j = x.order
        x.point = -1
        while True:
            if ((x.point == -1 or x.N == self.maximum_visit_count)
                    and np.any(x.V == -1) and np.any(x.board != -1)):
                self.make_leafs(x, j)
            x.V, x.N0 = self.back_up(j)
            if np.any(x.board != -1):
                arr0 = x.backup_calc(self.c_puct, self.maximum_visit_count)
                j0 = self.find_leaf(j, arr0)
                x.V0s = np.append(x.V0s, self.tree[j0].V0)
                x = self.tree[j0]
                x.N += 1
            j = x.order
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
                            resnet = self.resnet,
                            epsi = self.epsi,
                            visitcount=self.maximum_visit_count))

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

    def back_up(self, point):
        V = np.zeros(16)
        N = np.zeros(16)
        for i in self.tree:
            if i.point == point and np.any(i.board!=-1):        #查找所有合法叶节点
                V[i.num] = np.sum(i.V0s)
                N[i.num] = i.N
        return V, N

    def find_leaf(self, point, num):
        for i, x in enumerate(self.tree):
            if x.point == point and x.num == num:
                return i

def select_action(visit_counts, temperature):
    if temperature == 0:
        return np.argmax(visit_counts)
    adjusted_counts = visit_counts ** (1 / temperature)
    probs = adjusted_counts / np.sum(adjusted_counts)

    action = np.random.choice(len(visit_counts), p=probs)
    return action

def single_move(saved, title, total_visit_count):
    p_bar = tqdm.tqdm(total=total_visit_count - saved, desc=title)
    while True:
        N0 = tree.tree[0].N0
        tree.expand_and_evaluate()
        p_bar.update(1)
        p_bar.refresh()

        if  sum(N0) > total_visit_count:
            temp = 1 if tree.tree[0].ar < 30 else 0
            select_move = select_action(N0, temp)
            tree.restart(select_move)
            p_bar.close()
            return sum(tree.tree[0].N0)

def self_play(num, total_visit_count):
    saved_move = 0
    self_play_dict = []

    while True:
        tree_root = tree.tree[0]
        bd = tree_root.board
        if not np.any(bd==0):
            break
        saved_move = single_move(saved_move, "self_play_num:" + str(num), total_visit_count)
        pi = tree_root.N0 / sum(tree_root.N0)
        dict0 = {"board": tree_root.r_input.tolist(), "Pi": pi.tolist(), "P": 0}
        self_play_dict.append(dict0)
    score = score_count(self_play_dict)
    for i, x in enumerate(self_play_dict):
        x["P"] = int(score)
    name = time.asctime().replace(" ", "_")
    name = name.replace(":", "_")
    name += "of_model_" + str(num)
    with open("data/" + name + ".json", 'w') as f:
        f.write(json.dumps(self_play_dict, indent = 4))
    return len(self_play_dict)

def log3(x):
    x = int(x)
    fx = np.log(x)/np.log(3) if x > 0 else 0
    return np.ceil(fx)

def score_count(dict):
    input_ = [x["board"][0] for x in dict]
    end_input = np.array(input_[-1])
    end_board = end_input[1:-1, 1:-1]
    end_board = np.reshape(end_board, 16)
    raw_score = sum([score_from_num(x) for x in end_board])
    input_pool = np.array([x[0][1] for x in input_])
    input_pool = np.where(input_pool == 1, 0, input_pool)
    minus_score = sum([score_from_num(x) for x in input_pool])
    real_score = raw_score - minus_score
    return real_score

def score_from_num(num):
    if num == 0:
        return 0
    num_index = int(log3(num))
    index_f = np.array([3 ** x for x in range(num_index + 1)])
    index_b = index_f[::-1]
    score = sum(index_f * index_b)
    return score

def remove_duplicates(raw):
    data = []
    hashmap = []
    for i in raw:
        board = str(i["board"])
        board_hash = hash(board)
        if board_hash not in hashmap:
            data.append(i)
            hashmap.append(board_hash)
    return data

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
    data = remove_duplicates(data)
    return data

class TripleDataset(Dataset):
    def __init__(self):
        self.data = get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["board"]
        img = torch.tensor(img,dtype=torch.float32)
        target_pi = self.data[idx]["Pi"]
        target_pi = torch.tensor(target_pi,dtype=torch.float32)
        target_p = self.data[idx]["P"]
        target_p = torch.tensor([target_p],dtype=torch.float32)
        target_p = 1 / target_p
        target = torch.cat((target_pi, target_p), dim=0)
        return img, target

def train(dataset0):
    dataloader = DataLoader(batch_size=1, shuffle=True, dataset=dataset0)
    loss1 = MSELoss()
    loss2 = CrossEntropyLoss()
    loss1.cuda()
    loss2.cuda()
    optim = SGD(train_resnet.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
    #optim = Adam(train_resnet.parameters(), lr=0.001)
    epochs = 1
    for epoch in range(epochs):
        for data in tqdm.tqdm(dataloader):
            img, target = data
            img = img[0].cuda()
            target = target[0].cuda()

            outputs = train_resnet(img)
            result_loss1 = loss1(outputs[-1], target[-1])
            result_loss2 = loss2(outputs[:-1], torch.argmax(target[:-1]))

            result_loss = (result_loss1 + result_loss2)
            optim.zero_grad()
            result_loss.backward()
            optim.step()

if __name__ == "__main__":
    model_name = "demo"
    model_path = "model/"
    stat_path = "stat/"
    MAXIMUM_VISIT_COUNT = 5
    C_PUCT = 3.0
    EPSI = 0.25
    TOTAL_VISIT_COUNT = 800

    dirlist = os.listdir(stat_path)
    parser = argparse.ArgumentParser(description='Monte Carlo Tree Search Example')
    parser.add_argument('--mode', type=str, default='diverse_play')
    parser.add_argument('--selection', type=int, default = 0)
    args = parser.parse_args()
    all_data_num = [int(x.replace('demo','').replace('.json','')) for x in dirlist]
    all_data_num.append(-1)
    latest_num = max(all_data_num)
    if args.mode == 'diverse_play':
        index = latest_num
        while True:#predict
            index += 1
            resnet = ResNet()
            if model_name + str(index) + ".json" not in dirlist:
                with open(stat_path + model_name + str(index) + ".json", 'w') as f:
                    f.write(json.dumps([]))
                torch.save(resnet.state_dict(), model_path + model_name + str(index) + ".pt")
            with open(stat_path + model_name + str(index) + ".json", 'r') as f:
                move_steps_list = json.loads(f.read())
            if model_name + str(index) + ".pt" in os.listdir(model_path):
                print(model_name + str(index) + " has been loaded")
                resnet.load_state_dict(torch.load(model_path + model_name + str(index) + ".pt"))
            resnet = resnet.cuda()
            resnet = resnet.half()
            tree = Tree(resnet, maximum_visit_count = MAXIMUM_VISIT_COUNT, c_puct = C_PUCT, epsi = EPSI)
            move_steps = self_play(index, TOTAL_VISIT_COUNT)
            move_steps_list.append(move_steps)
            with open(stat_path + model_name + str(index) + ".json", 'w') as f:
                f.write(json.dumps(move_steps_list))
    if args.mode == 'single_play':
        index = args.selection
        resnet = ResNet()
        if model_name + str(index) + ".json" not in dirlist:
            with open(stat_path + model_name + str(index) + ".json", 'w') as f:
                f.write(json.dumps([]))
            torch.save(resnet.state_dict(), model_path + model_name + str(index) + ".pt")
        while True:
            with open(stat_path + model_name + str(index) + ".json", 'r') as f:
                move_steps_list = json.loads(f.read())
            if model_name + str(index) + ".pt" in os.listdir(model_path):
                print(model_name + str(index) + " has been loaded")
                resnet.load_state_dict(torch.load(model_path + model_name + str(index) + ".pt"))
            resnet = resnet.cuda()
            resnet = resnet.half()
            tree = Tree(resnet, maximum_visit_count=MAXIMUM_VISIT_COUNT, c_puct=C_PUCT, epsi=EPSI)
            move_steps = self_play(index, TOTAL_VISIT_COUNT)
            move_steps_list.append(move_steps)
            with open(stat_path + model_name + str(index) + ".json", 'w') as f:
                f.write(json.dumps(move_steps_list))
    if args.mode == 'train':
        index = args.selection
        dataset = TripleDataset()
        train_resnet = ResNet()
        train_resnet.load_state_dict(torch.load(model_path + model_name + str(index) + ".pt"))
        train_resnet = train_resnet.cuda()
        train_resnet = train_resnet.to(dtype=torch.float32)
        train(dataset)
        torch.save(train_resnet.state_dict(),model_path + model_name + str(index) + ".pt")