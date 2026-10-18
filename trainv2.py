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
from model import ResNet
import copy

class Tree:
    def __init__(self, resnet, device_id):
        super(Tree, self).__init__()
        self.device_id = device_id
        self.tree = []
        self.resnet = resnet
        self.c_puct = C_PUCT
        self.gm = Triple()
        torch.manual_seed(int(time.time()))
        #self.pool = torch.randint(10,99,(2,)) random incoming block
        init_board = torch.tensor([[0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0]],dtype=torch.float16)
        init_pool = torch.randint(10,99,(2,))
        init_placement = torch.full((2, ), -1)
        node = {
            "index": 0,
            "ar": 0,
            "pre_index": -1,
            "raw_board": init_board,
            "pool": init_pool,
            "feather_planes": torch.zeros((7,6,6), dtype=torch.float16),
            "N" : 0,
            "P" : torch.zeros((16,), dtype=torch.float16),
            "placements": init_placement,
            "suf_index": torch.full((16,), -1, dtype=torch.int16),
            "p0": torch.zeros((16,16), dtype=torch.float16),
            "v0": torch.zeros((16,), dtype=torch.float16),
            "r_input": torch.zeros((8, 6, 6)),
            "num": 0,
            "N0": torch.zeros((16, ), dtype=torch.float16),
            "N_s": 0,
            "Q1": torch.tensor([]),
            "Q": torch.zeros((16, ), dtype=torch.float16),
            "puct": torch.tensor([]),
            "U": torch.tensor([]),
            "num_pool": torch.tensor([]),
            "W": torch.zeros((16,), dtype=torch.float16),
            "raw_sliced": torch.ones((16,), dtype=torch.float16)
        }
        self.node_pattern = copy.deepcopy(node)
        self.tree.append(node)
        self.node_init(0)
        output = self.forward([self.tree[0]])
        self.tree[0]["P"] = output[0][:-1]
        self.tree[0]["V0"] = output[0][-1]
        
    def node_init(self, node_index):
        node = self.tree[node_index]
        
        placement = node["placements"]
        
        self.gm.board = node["raw_board"].clone()
        self.gm.board = torch.flatten(self.gm.board)
        
        if placement[0] >= 0:
            self.gm.place(placement[0], placement[1])
            #self.gm.update_pool() #not required
        if placement[0] == -2:
            self.gm.board = torch.full((16,), -1)
        node["raw_board"] = self.gm.board.clone()
        node["raw_board"] = torch.reshape(node["raw_board"], (4, 4))
        
        possible_num = self.gm.possible_num()
        node["num_pool"] = torch.tensor([possible_num[x % len(possible_num)] for x in node["pool"]])
        
        node["V0"] = torch.tensor(0.)
        node['P'] = torch.zeros(16)

        node["N0"] = torch.zeros(16, dtype=torch.int64)
        node["W"] = torch.zeros(16, dtype=torch.float32)
        node["Q"] = torch.zeros(16, dtype=torch.float32)
        
        if torch.any(node["raw_board"] != -1):
            input1 = F.pad(node["raw_board"],(1,1,1,1),mode="constant",value=0)
            input2 = F.pad(node["num_pool"],(0, 2),mode="constant", value=0)
            input2 = torch.reshape(input2,(2,2))
            input2 = F.pad(input2,(0,4,0,4), mode="constant", value=0)
            input = input1 + input2
            input = input.to(torch.float16)
            input = torch.reshape(input,(1,6,6))
            #self.input = input.to(torch.float16)
            input = input.to(self.device_id)
            feather_planes = node["feather_planes"].to(self.device_id)
            node["r_input"] = torch.cat((input, feather_planes),0)
            node["r_input"] = node["r_input"].to(torch.float16)
        
            

    def select(self):
        #print(self.tree[-1].board)
        x = self.tree[0]
        j = x['index']
        x['pre_index'] = -1
        expanded = False
        while True:
            if (x["N"] == 0 and torch.any(x["raw_board"] != -1)):
                self.expand_and_evaluate(x, j)
                expanded = True
            suf_nodes_index = x["suf_index"]
            
            if torch.any(x["raw_board"] != -1):
                arr0 = self.select_calc(x["index"], self.c_puct)
                leaf_index = suf_nodes_index[arr0]
                j0 = leaf_index
                x = self.tree[j0]
            j = x['index']
            if expanded or torch.any(x["raw_board"] == -1):
                break
        self.back_up(j)
        
            
    def select_calc(self, node_index, c):
        node = self.tree[node_index]
        node["Q"] = node['W'] / node["N0"]
        node["Q"] = torch.nan_to_num(node["Q"], nan=0)
        node["puct"] = (node["N"] ** (1/2)    ) / (torch.ones(16) + node["N0"])
        node["U"] = c * node["P"] * node["puct"]
        minus_inf = torch.tensor([-torch.inf if i == 0 else 0 for i in node["raw_sliced"]])

        return torch.argmax(node["U"] + node["Q"] + minus_inf)

    def make_leafs_prop(self, node):
        board_1 = torch.reshape(node["raw_board"], (16, ))
        raw_sliced = 1 - torch.tensor([min(x,1) for x in board_1])
        placement_list = []
        for i, x in enumerate(raw_sliced):
            placement = -2, -2, node["raw_board"]
            if x != 0:
                placement = i, node["num_pool"][0], node["raw_board"]
            placement_list.append(placement)
        
        return placement_list, raw_sliced

    def expand_and_evaluate(self, father, point):
        prop, raw_sliced = self.make_leafs_prop(father)
        ar = father['ar']
        init_pool = torch.randint(10,99,(2,))
        feather_planes = father['r_input'][:-1]
        suf_index = torch.zeros((16,), dtype=torch.int16)
        
        self.tree[father["index"]]["P"] = self.tree[father["index"]]["P"] * raw_sliced
        self.tree[father["index"]]["raw_sliced"] = raw_sliced
        
        for i,x in enumerate(prop):
            node_ = copy.deepcopy(self.node_pattern)
            
            node_["placements"] = x[:2]
            node_["pool"] = init_pool
            node_["pre_index"] = point
            node_["num"] = i
            node_["ar"] = ar + 1
            node_["raw_board"] = x[2]
            node_["feather_planes"] = feather_planes
            node_["index"] = len(self.tree)
            
            suf_index[i] = node_["index"]
            
            self.tree.append(node_)
            self.node_init(node_["index"])
        
        self.tree[father["index"]]["suf_index"] = suf_index
        
        output_ = self.forward(self.tree[-len(prop):])
        for i in range(1, len(prop) + 1):
            self.tree[-i]["P"] = output_[-i][:-1]
            self.tree[-i]["V0"] = output_[-i][-1]

    def restart(self, select_calc):
        temp_tree = []
        list0 = [self.tree[select_calc+1]["index"]]
        mapping = [] #(raw_index, new_index)
        while True:
            for i, x in enumerate(list0):
                if x != 0:
                    father_node = self.tree[x]
                    suf_indexes = father_node["suf_index"]
                    new_list = [self.tree[j]['index'] for j in suf_indexes if torch.any(suf_indexes != -1) ] #node
                    list0[i] = 0
                    list0 += new_list
                    node0 = self.tree[x]
                    temp_tree.append(node0)
                    mapping.append(node0["index"])
            if not any(list0):
                break
        mapping.append(0)
        for i in temp_tree:
            raw_index = i["index"]
            raw_pre_index = i["pre_index"]
            raw_suf_index = i["suf_index"]
            
            i["index"] = mapping.index(raw_index)
            i['pre_index'] = mapping.index(raw_pre_index)
            
            suf_index_map = torch.tensor([mapping.index(int(x)) for x in raw_suf_index]) if torch.any(raw_suf_index != -1) else torch.full((16,), -1, dtype=torch.int16)
            i['suf_index'] = suf_index_map
        temp_tree[0]["pre_index"] = -1
        self.tree = temp_tree

    def back_up(self, leaf_index):
        leaf_node = self.tree[leaf_index]
        leaf_v = leaf_node["V0"]
        #leaf_node["N"] += 1 #update leaf node 
        
        node_num = leaf_node["num"]
        j = leaf_node["pre_index"] #initialized by pre leaf node index
        
        while j != -1:
            node = self.tree[j]
            node["N"] += 1 #N update
            node["N0"][node_num] += 1 #update N0
            node["W"][node_num] += leaf_v # update W
            
            j = node['pre_index']
            node_num = node["num"]
                 
    def forward(self, nodes):
        forward_list = []
        batch_input_ = []
        ar = nodes[0]["ar"]
        
        output_list = [torch.zeros(17, ) for _ in range(len(nodes))]
        for index, node in enumerate(nodes):
            if torch.any(node["raw_board"] != -1):
                forward_list.append(index)
                batch_input_.append(node["r_input"])
        if len(batch_input_) == 0:
            return output_list
        batch_input = torch.stack([i for i in batch_input_])
        batch_input_ = batch_input.to(self.device_id)
        with torch.no_grad():
            self.resnet.eval()
            self.outputs = self.resnet(batch_input_)
            self.outputs = self.outputs.cpu()

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
        img, score_now = self.feather_planes_gen(click_history, grid_history)
        img = torch.tensor(img,dtype=torch.float32)
        target_pi = self.data[idx]["pi_history"]
        target_pi = torch.tensor(target_pi,dtype=torch.float32)
        final_P = self.data[idx]["final_P"]
        target_P = (final_P - score_now) / CO_P
        target_P = torch.tensor([target_P],dtype=torch.float32)
        target = torch.cat((target_pi, target_P), dim=0)
        return img, target  

    def feather_planes_gen(self, click_history, grid_history):
        gen_game = Triple()
        last_feather_plane = torch.zeros((7, 6, 6), dtype=torch.int16)
        placement = zip(click_history, grid_history)
        feather_planes = []
        for index, value in placement:
            gen_game.num_pool = value
            input1 = F.pad(gen_game.board,(1,1,1,1),mode="constant",value=0)
            input2 = F.pad(gen_game.num_pool,(0, 2),mode="constant", value=0)
            input2 = torch.reshape(input2,(2,2))
            input2 = F.pad(input2,(0,4,0,4), mode="constant", value=0)
            input = input1 + input2
            input = input.to(torch.float16)
            input = torch.reshape(input,(1,6,6))
            feather_plane = torch.cat(input, last_feather_plane, 0)
            last_feather_plane = feather_plane[:-1]
            feather_planes.append(feather_plane)

            gen_game.place(index, value[0])
            score_now = gen_game.score
        
        return feather_planes, score_now
    
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
                    data.append(json.loads(f.read()))
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
            img = img.cuda()
            target = target.cuda()

            outputs = train_resnet(img)
            result_loss1 = loss1(outputs[:, 16], target[:,16])
            result_loss2 = loss2(outputs[:, :16], target[:,:16])

            result_loss = (result_loss1 + result_loss2)
            optim.zero_grad()
            result_loss.backward()
            optim.step()

class AlphaTriple:
    def __init__(self, device_list, device_index, model_real_path):
        self.device_list = device_list
        self.deivce_index = device_index
        self.model_real_path = model_real_path
        
        if device_list == torch.device("cpu"):
            self.device = device_list
        else:
            self.device = self.device_list[self.deivce_index]
        self.resnet = ResNet()

        if self.model_real_path is not None:
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
            N0 = self.tree.tree[0]['N0']
            root_P = self.tree.tree[0]["P"]

            if self.tree.tree[0]["ar"] > EXPLORATION_THRESHOLD:
                dirichlet_dist  = torch.distributions.Dirichlet(ALPHA)
                eta = dirichlet_dist.sample()
                root_P = (1 - EPSI) * root_P + EPSI * eta

            self.tree.select()
            with tqdm.tqdm.get_lock():
                p_bar.update(1)
                if  sum(N0) > TOTAL_VISIT_COUNT:
                    temp = 1 if self.tree.tree[0]['ar'] < EXPLORATION_THRESHOLD else 0
                    select_move = self.select_action(N0, temp)
                    self.tree.restart(select_move)
                    p_bar.close()
                    return int(sum(self.tree.tree[0]['N0'])), select_move
        
    def self_play(self):
        saved_move = 0
        click_history = []
        pi_history = []
        board_history = []
        grid_history = []
        while True:
            tree_root = self.tree.tree[0]
            bd = tree_root['raw_board']
            if not torch.any(bd==0):
                break
            saved_move, move = self.single_move(saved_move)
            pi = tree_root["N0"] / sum(tree_root["N0"])
            #dict0 = {"board": tree_root["r_input"].tolist(), "Pi": pi.tolist(), "P": 0}
            click_history.append(int(move))
            grid_history.append(tree_root["num_pool"].tolist())
            pi_history.append(pi.tolist())
            board_history.append(bd.tolist())
            #print(bd)
            
        experience = {
                "final_board": board_history[-1],
                "click_history": click_history,
                "grid_history": grid_history,
                "pi_history": pi_history,
                "steps": len(click_history),
                "final_P":0
        }
        score = score_count(experience)
        experience["final_P"] = int(score)
        name = time.asctime().replace(" ", "_")
        name = name.replace(":", "_")
        name += "of_model_" + str(self.model_index)
        with open("data/" + name + ".json", 'w') as f:
            f.write(json.dumps(experience, indent = 4))

def worker_init_fn(worker_id):
    seed = int(time.time() * 1000) % (10**9) + os.getpid()
    torch.manual_seed(seed)

def run(arg):
    steps, device_list, device_index, model_real_path= arg
    for i in range(steps):
        alphatriple = AlphaTriple(device_list=device_list, device_index=device_index, model_real_path=model_real_path)
        alphatriple.self_play()

def generator(epochs, device_count, model_real_path):#EPOCH = TASK_PER_GPU * DEVICE_COUNT * STEPS_PER_DEVICE
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
        tasks = [(steps_per_device, device_list, i, model_real_path) for i in range(device_count * TASK_PER_DEVICE)]
        with multiprocessing.Pool(processes=device_count * TASK_PER_DEVICE, initializer=worker_init_fn, initargs=(1,)) as po:
            po.map(run, tasks)
        
    else: #单进程
        run((epochs, device_list, 0, model_real_path))  

C_PUCT = 3.0
EPSI = 0.25
CO_P = 1000
TOTAL_VISIT_COUNT = 800
TASK_PER_DEVICE = 1
MAXIMUM_DEVICE_COUNT = 1
EXPLORATION_THRESHOLD = 30
ALPHA = torch.full((16,), 0.03)

if __name__ == "__main__":
    model_name = "demo"
    model_path = "model/"
    
    
    device_count = torch.cuda.device_count()
    device_count = min(device_count, MAXIMUM_DEVICE_COUNT)

    parser = argparse.ArgumentParser(description='Monte Carlo Tree Search Example')
    parser.add_argument('--mode', type=str, default='generate')
    parser.add_argument('--selection', type=int, default = 0)
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()
    if args.mode == "generate":
        
        model_index = args.selection
        generate_epoch = args.epoch
        model_real_path = model_path + model_name + str(model_index) + ".pt" if model_index > 0 else None
        generator(generate_epoch, device_count, model_real_path)
        
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
    if args.mode == "generate_empty_model":
        index = args.selection
        train_resnet = ResNet()
        train_resnet = train_resnet.to(dtype=torch.float32)
        torch.save(train_resnet.state_dict(),model_path + model_name + str(index) + ".pt")
            