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
import multiprocessing

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
        length = 40  
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
    def __init__(self,  pool, board, feather_planes, resnet, visitcount, device_id, epsi = 0.25,
                point = 0, num = 0, ar = 0, placement = np.full(2,-1), ):
        super(Node,self).__init__()
        self.device_id = device_id
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
            gm.update_pool()
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
            #self.input = input.to(torch.float16)
            input = input.to(self.device_id)
            feather_planes = feather_planes.to(self.device_id)
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
    def __init__(self, resnet, device_id):
        super(Tree, self).__init__()
        self.device_id = device_id
        self.tree = []
        self.resnet = resnet
        self.maximum_visit_count = MAXIMUM_VISIT_COUNT
        self.c_puct = C_PUCT
        self.epsi = EPSI

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
                            visitcount = self.maximum_visit_count,
                            device_id=self.device_id
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
                            visitcount=self.maximum_visit_count,
                            device_id=self.device_id))

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

def score_count(experience):
    count_score_gm = Triple()
    placement = zip(experience.click_history, experience.grid_history)
    for index, value in placement:
        count_score_gm.place(index, value[0])
    return count_score_gm.score



class TripleDataset(Dataset):
    def __init__(self):
        self.data = self.get_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        click_history = self.data[idx]["click_history"]
        grid_history = self.data[idx]["grid_history"]
        img = self.feather_planes_gen(click_history, grid_history)
        img = torch.tensor(img,dtype=torch.float32)
        target_pi = self.data[idx]["Pi"]
        target_pi = torch.tensor(target_pi,dtype=torch.float32)
        target_p = self.data[idx]["P"]
        target_p = torch.tensor([target_p],dtype=torch.float32)
        target_p = 1 / target_p
        target = torch.cat((target_pi, target_p), dim=0)
        return img, target

    def feather_planes_gen(self, click_history, grid_history):
        gen_game = Triple()
        last_feather_plane = np.zeros((7, 6, 6), dtype=np.int16)
        placement = zip(click_history, grid_history)
        feather_planes = []
        for index, value in placement:
            gen_game.num_pool = value
            gen_game.place(index, value[0])
            input1 = np.pad(gen_game.board,pad_width=1,mode="constant",constant_values=0)
            input2 = np.pad(gen_game.num_pool,(0, 2),mode="constant",constant_values=0)
            input2 = np.reshape(input2,(2,2))
            input2 = np.pad(input2,((0,4),(0,4)),mode="constant",constant_values=0)
            input = torch.tensor(input1 + input2,dtype=torch.float16)
            input = torch.reshape(input,(1,6,6))
            feather_plane = torch.cat(input, last_feather_plane, 0)
            last_feather_plane = feather_plane[:-1]
            feather_planes.append(feather_plane)
        return feather_planes
    
    def remove_duplicates(self, raw):
        data = []
        hashmap = []
        for i in raw:
            board = str(i["board"])
            board_hash = hash(board)
            if board_hash not in hashmap:
                data.append(i)
                hashmap.append(board_hash)
        return data

    def get_data(self):
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
        data = self.remove_duplicates(data)
        return data

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

class AlphaTriple:
    def __init__(self, device_, model_path = None):
        device_list, self.deivce_index = device_
        device = device_list[self.deivce_index]
        
        self.resnet = ResNet()
        if model_path is not None:
            self.resnet.load_state_dict(torch.load(model_path))
        self.resnet = self.resnet.to(device)
        self.resnet = self.resnet.half()
        self.tree = Tree(self.resnet, device_id=device)
        self.device_name = str(device)
        
    def select_action(self, visit_counts, temperature):
        if temperature == 0:
            return np.argmax(visit_counts)
        adjusted_counts = visit_counts ** (1 / temperature)
        probs = adjusted_counts / np.sum(adjusted_counts)

        action = np.random.choice(len(visit_counts), p=probs)
        return action

    def single_move(self, saved, title):
        p_bar = tqdm.tqdm(total=TOTAL_VISIT_COUNT - saved, desc=title+ " on " + self.device_name, position=self.deivce_index)
        while True:
            N0 = self.tree.tree[0].N0
            self.tree.expand_and_evaluate()
            p_bar.update(1)
            p_bar.refresh()

            if  sum(N0) > TOTAL_VISIT_COUNT:
                temp = 1 if self.tree.tree[0].ar < 30 else 0
                select_move = self.select_action(N0, temp)
                self.tree.restart(select_move)
                p_bar.close()
                return sum(self.tree.tree[0].N0), select_move
    
    def self_play(self, num = 0):
        saved_move = 0
        click_history = []
        pi_history = []
        board_history = []
        while True:
            tree_root = self.tree.tree[0]
            bd = tree_root.board
            if not np.any(bd==0):
                break
            saved_move, move = self.single_move(saved_move, "self_play_num:" + str(num), TOTAL_VISIT_COUNT)
            pi = tree_root.N0 / sum(tree_root.N0)
            #dict0 = {"board": tree_root.r_input.tolist(), "Pi": pi.tolist(), "P": 0}
            click_history.append(move)
            grid_history = tree_root.num_pool.tolist()
            pi_history.append(pi.tolist())
            board_history.append(bd.tolist())
            
        experience = {
                "final_board": board_history[-1],
                "click_history": click_history,
                "grid_history": grid_history,
                "pi_history": pi_history,
                "steps": len(click_history),
                "P":0
        }
        score = score_count(experience)
        for i, x in enumerate(experience):
            x["P"] = int(score)
        name = time.asctime().replace(" ", "_")
        name = name.replace(":", "_")
        name += "of_model_" + str(num)
        with open("data/" + name + ".json", 'w') as f:
            f.write(json.dumps(experience, indent = 4))

def run(device, model_id):
    alphatriple = AlphaTriple(device, model_path=model_id)
    alphatriple.self_play(model_id)

def generator(device_count, epochs, model_id):
    if device_count != 0:
        device_list = [torch.device(f"cuda:{i}") for i in range(device_count)]
        print(f"Using {device_count} GPUs.")
    else:
        device_list = torch.device("cpu")
        print("No GPU found, using CPU.")
        device_count = 1
    po = multiprocessing.Pool(device_count)
    for i in range(0, epochs):
        po.apply_async(run, args=((device_list, i % device_count), model_id))
    po.close()
    po.join()
    
if __name__ == "__main__":
    model_name = "demo"
    model_path = "model/"
    MAXIMUM_VISIT_COUNT = 5
    C_PUCT = 3.0
    EPSI = 0.25
    TOTAL_VISIT_COUNT = 800
    
    device_count = torch.cuda.device_count()
    
    parser = argparse.ArgumentParser(description='Monte Carlo Tree Search Example')
    parser.add_argument('--mode', type=str, default='diverse_play')
    parser.add_argument('--selection', type=int, default = 0)
    parser.add_argument('--epoch', type=int, default=1)
    args = parser.parse_args()
    if args.mode == "generate":
        index = args.selection
        generate_epoch = args.epoch
        model_real_path = model_path + model_name + str(index) + ".pt" if index > 0 else None
        generator(device_count, generate_epoch, model_real_path)
        
    if args.mode == 'train':
        index = args.selection
        training_epoch = args.epoch
        dataset = TripleDataset()
        train_resnet = ResNet()
        train_resnet.load_state_dict(torch.load(model_path + model_name + str(index) + ".pt"))
        train_resnet = train_resnet.cuda()
        train_resnet = train_resnet.to(dtype=torch.float32)
        train(dataset)
        torch.save(train_resnet.state_dict(),model_path + model_name + str(index) + ".pt")