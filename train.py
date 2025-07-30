import torch
from torch import nn
from torch.nn import Sequential,Conv2d,Linear,Flatten,ReLU,Sigmoid,BatchNorm2d,MSELoss,CrossEntropyLoss,Softmax
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD
from Triple import Triple
import time
import tqdm
import json, os, argparse
import multiprocessing
import torch.nn.functional as F

class BasicNet(nn.Module):
    def __init__(self, channel):
        super(BasicNet, self).__init__()
        self.block1 = Sequential(
            Conv2d(channel, channel, 3, stride=1, padding=1),
            BatchNorm2d(channel),
            ReLU(),
            Conv2d(channel, channel, 3, stride=1, padding=1),
            BatchNorm2d(channel)
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
            BatchNorm2d(2),
            ReLU(),
            Flatten(1),
            Linear(2 * 16, 16),
            Softmax(dim=1)
        )
    def forward(self, x):
        x = self.net(x)
        return x

class value_head(nn.Module):
    def __init__(self, channel):
        super(value_head, self).__init__()
        self.net = Sequential(
            Conv2d(channel, 1, 1, stride=1),
            BatchNorm2d(1),
            ReLU(),
            Flatten(1),
            Linear(16, 256),
            ReLU(),
            Linear(256, 1),
            Sigmoid()
        )
    def forward(self, x):
        x = self.net(x)
        return x

class ResNet(nn.Module):
    def __init__(self, channels = 256, length = 40):
        super(ResNet, self).__init__()
        self.Convolutional = Sequential(
            Conv2d(8, channels, 3, stride=1, padding=0),
            BatchNorm2d(channels),
            ReLU()
        )
        self.net = Sequential(*[BasicNet(channels) for _ in range(length)])
        self.policy = policy_head(channels)
        self.value = value_head(channels)

    def forward(self, x):
        x = self.Convolutional(x)
        x = self.net(x)
        p = self.policy(x)
        v = self.value(x)
        output = torch.cat((p, v), dim=1)
        return output

class Node:
    def __init__(self,  pool, board, feather_planes, visitcount, device_id,
                point = 0, num = 0, ar = 0, placement = torch.full((2,),-1), ):
        super(Node, self).__init__()
        self.device_id = device_id
        self.placement = placement
        self.pool = pool
        self.point = point
        self.ar = ar
        self.N = 0
        self.num = num
        self.V = torch.full((16,),-1)
        self.N0 = torch.tensor([])
        self.board = board
        self.order = time.time()
        self.order += 100000 * torch.rand(1).item()
        self.feather_planes = feather_planes
        gm = Triple()
        gm.board = self.board.clone()
        gm.board = torch.flatten(gm.board)

        if placement[0] >= 0:
            gm.place(placement[0], placement[1])
            gm.update_pool()
        if placement[0] == -2:
            gm.board = torch.full((16,), -1)
        self.board = gm.board.clone()
        self.board = torch.reshape(self.board, (4, 4))

        possible_num = gm.possible_num()
        self.num_pool = torch.tensor([possible_num[x % len(possible_num)] for x in pool])
        
        self.V0 = torch.tensor(0.)
        self.P = torch.zeros(16)
        if torch.any(self.board!=-1):
            input1 = F.pad(self.board,(1,1,1,1),mode="constant",value=0)
            input2 = F.pad(self.num_pool,(0, 2),mode="constant", value=0)
            input2 = torch.reshape(input2,(2,2))
            input2 = F.pad(input2,(0,4,0,4), mode="constant", value=0)
            input = input1 + input2
            input = input.to(torch.float16)
            input = torch.reshape(input,(1,6,6))
            #self.input = input.to(torch.float16)
            input = input.to(self.device_id)
            feather_planes = feather_planes.to(self.device_id)
            self.r_input = torch.cat((input, feather_planes),0)
            self.r_input = self.r_input.to(torch.float16)
            
    def backup_calc(self, c, visit_count):
        self.N_s = sum(self.N0)
        self.Q1 = self.V / (self.N_s + visit_count)
        self.Q = self.Q1 / max(self.Q1)
        self.puct = (self.N_s ** 1/2) / (torch.ones(16) + self.N0)
        self.U = c * self.P * self.puct

        return torch.argmax(self.U + self.Q)

    def make_leafs_prop(self):
        board_1 = torch.reshape(self.board,(16,))
        raw_sliced = 1 - torch.tensor([min(x,1) for x in board_1])
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
        torch.manual_seed(int(time.time()))
        self.pool = torch.randint(10,99,(3000,))
        init_board = torch.tensor([[0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0]],dtype=torch.float16)
        init_pool = self.pool[:2]
        feather_planes = torch.zeros((7,6,6))

        self.tree.append(Node(board = init_board,
                            pool = init_pool,
                            point=-1,
                            feather_planes = feather_planes,
                            visitcount = self.maximum_visit_count,
                            device_id=self.device_id
                            ))
        output = self.forward([self.tree[0]])
        self.tree[0].P = output[0][:-1]
        self.tree[0].V0 = output[0][-1]
        self.tree[0].V0s = torch.full((self.maximum_visit_count,), self.tree[0].V0)

    def expand_and_evaluate(self):
        #print(self.tree[-1].board)
        x = self.tree[0]
        j = x.order
        x.point = -1
        while True:
            if ((x.point == -1 or x.N == self.maximum_visit_count)
                    and torch.any(x.V == -1) and torch.any(x.board != -1)):
                self.make_leafs(x, j)
            x.V, x.N0 = self.back_up(j)
            if torch.any(x.board != -1):
                arr0 = x.backup_calc(self.c_puct, self.maximum_visit_count)
                j0 = self.find_leaf(j, arr0)
                oned_V0 = torch.reshape(self.tree[j0].V0,(1,))
                x.V0s = torch.cat((x.V0s, oned_V0), dim=0)
                x = self.tree[j0]
                x.N += 1
            j = x.order
            if (x.N < self.maximum_visit_count
                    or torch.any(x.board == -1)):
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
                            visitcount=self.maximum_visit_count,
                            device_id=self.device_id))
        output_ = self.forward(self.tree[-len(prop):])
        for i in range(1, len(prop) + 1):
            self.tree[-i].P = output_[-i][:-1]
            self.tree[-i].V0 = output_[-i][-1]
            self.tree[-i].V0s = torch.full((self.maximum_visit_count,), self.tree[-i].V0)
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
        V = torch.zeros(16)
        N = torch.zeros(16)
        for i in self.tree:
            if i.point == point and torch.any(i.board!=-1):        #查找所有合法叶节点
                V[i.num] = sum(i.V0s)
                N[i.num] = i.N
        return V, N

    def find_leaf(self, point, num):
        for i, x in enumerate(self.tree):
            if x.point == point and x.num == num:
                return i
    
    def forward(self, nodes):
        forward_list = []
        batch_input_ = []
        ar = nodes[0].ar
        alpha = torch.full((16,), 0.03)
        output_list = [torch.zeros(17, ) for _ in range(len(nodes))]
        for index, node in enumerate(nodes):
            if torch.any(node.board!=-1):
                forward_list.append(index)
                batch_input_.append(node.r_input)
        if len(batch_input_) == 0:
            return output_list
        batch_input = torch.stack([i for i in batch_input_])
        batch_input_ = batch_input.to(self.device_id)
        with torch.no_grad():
            self.resnet.eval()
            self.outputs = self.resnet(batch_input_)
            self.outputs = self.outputs.cpu()
        for output in self.outputs:
            if ar >= 30:
                dirichlet_dist  = torch.distributions.Dirichlet(alpha)
                eta = dirichlet_dist.sample()
                output[:-1] = (1 - self.epsi) * output[:-1] + self.epsi * eta
            output[-1] = 1 / output[-1]
        index_ = 0
        for index, output in enumerate(output_list):
            if index in forward_list:
                output[:-1] = self.outputs[index_][:-1]
                output[-1] = self.outputs[index_][-1]
                index_ += 1
        return output_list

def score_count(experience):
    count_score_gm = Triple()
    placement = zip(experience["click_history"], experience["grid_history"])
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
        last_feather_plane = torch.zeros((7, 6, 6), dtype=torch.int16)
        placement = zip(click_history, grid_history)
        feather_planes = []
        for index, value in placement:
            gen_game.num_pool = value
            gen_game.place(index, value[0])
            input1 = F.pad(self.board,(1,1,1,1),mode="constant",value=0)
            input2 = F.pad(self.num_pool,(0, 2),mode="constant", value=0)
            input2 = torch.reshape(input2,(2,2))
            input2 = F.pad(input2,(0,4,0,4), mode="constant", value=0)
            input = input1 + input2
            input = input.to(torch.float16)
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
    def __init__(self, device_list, device_index):
        self.device_list = device_list
        self.deivce_index = device_index
        self.model_index = model_index
        self.device = self.device_list[self.deivce_index]
        self.resnet = ResNet()
        self.model_path = model_real_path
        if self.model_path is not None:
            self.resnet.load_state_dict(torch.load(self.model_path))

        self.resnet = self.resnet.to(self.device)
        self.resnet = self.resnet.half()
        self.tree = Tree(self.resnet, device_id=self.device)
        self.device_name = str(self.device)
        
    def select_action(self, visit_counts, temperature):
        if temperature == 0:
            return torch.argmax(visit_counts)
        adjusted_counts = visit_counts ** (1 / temperature)
        probs = adjusted_counts / sum(adjusted_counts)
        #print(sum(probs))
        action = torch.multinomial(probs, num_samples=1, replacement=True)
        #print(probs, action)
        return action

    def single_move(self, saved):
        p_bar = tqdm.tqdm(total=TOTAL_VISIT_COUNT - saved, desc="self_play_on_" + self.device_name, 
                          position=self.deivce_index, leave=False)
        while True:
            N0 = self.tree.tree[0].N0
            self.tree.expand_and_evaluate()
            with tqdm.tqdm.get_lock():
                p_bar.update(1)
                if  sum(N0) > TOTAL_VISIT_COUNT:
                    temp = 1 if self.tree.tree[0].ar < 30 else 0
                    select_move = self.select_action(N0, temp)
                    self.tree.restart(select_move)
                    p_bar.close()
                    return int(sum(self.tree.tree[0].N0)), select_move
        
    def self_play(self):
        saved_move = 0
        click_history = []
        pi_history = []
        board_history = []
        grid_history = []
        while True:
            tree_root = self.tree.tree[0]
            bd = tree_root.board
            if not torch.any(bd==0):
                break
            saved_move, move = self.single_move(saved_move)
            pi = tree_root.N0 / sum(tree_root.N0)
            #dict0 = {"board": tree_root.r_input.tolist(), "Pi": pi.tolist(), "P": 0}
            click_history.append(int(move))
            grid_history.append(tree_root.num_pool.tolist())
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
        experience["P"] = int(score)
        name = time.asctime().replace(" ", "_")
        name = name.replace(":", "_")
        name += "of_model_" + str(self.model_index)
        with open("data/" + name + ".json", 'w') as f:
            f.write(json.dumps(experience, indent = 4))

def worker_init_fn(worker_id):
    seed = int(time.time() * 1000) % (10**9) + os.getpid()
    torch.manual_seed(seed)

def run(arg):
    steps, device_list, device_index = arg
    for i in range(steps):
        alphatriple = AlphaTriple(device_list=device_list, device_index=device_index)
        alphatriple.self_play()

def generator(epochs, device_count):#EPOCH = TASK_PER_GPU * DEVICE_COUNT * STEPS_PER_DEVICE
    if device_count != 0:
        device_list = [torch.device(f"cuda:{i}") for i in range(device_count)]
        print(f"Using {device_count} GPUs.")
        device_list = device_list * TASK_PER_DEVICE
    else:
        device_list = torch.device("cpu")
        print("No GPU found, using CPU.")
        device_count = 1
    if device_count > 1: #多进程
        steps_per_device = epochs // device_count // TASK_PER_DEVICE
        tasks = [(steps_per_device, device_list, i) for i in range(device_count * TASK_PER_DEVICE)]
        with multiprocessing.Pool(processes=device_count * TASK_PER_DEVICE, initializer=worker_init_fn, initargs=(1,)) as po:
            po.map(run, tasks)
        
    else: #单进程
        run((epochs, device_list, 0))  
    
if __name__ == "__main__":
    model_name = "demo"
    model_path = "model/"
    MAXIMUM_VISIT_COUNT = 5
    C_PUCT = 3.0
    EPSI = 0.25
    TOTAL_VISIT_COUNT = 800
    TASK_PER_DEVICE = 1
    
    device_count = torch.cuda.device_count()

    parser = argparse.ArgumentParser(description='Monte Carlo Tree Search Example')
    parser.add_argument('--mode', type=str, default='diverse_play')
    parser.add_argument('--selection', type=int, default = 0)
    parser.add_argument('--epoch', type=int, default=1)
    args = parser.parse_args()
    if args.mode == "generate":
        
        model_index = args.selection
        generate_epoch = args.epoch
        model_real_path = model_path + model_name + str(model_index) + ".pt" if model_index > 0 else None
        generator(generate_epoch, device_count)
        
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