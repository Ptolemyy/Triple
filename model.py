from multiprocessing import pool

import torch
from torch import nn
from torch.nn import Sequential,Conv2d,Linear,Flatten,ReLU,Sigmoid,BatchNorm2d,MSELoss,CrossEntropyLoss,Softmax
import multiprocessing
import queue
import os

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
    
class Backend():
    def __init__(self, model_path, dev_count, batch_size):
        ctx = multiprocessing.get_context("spawn")
        self.task_queue = ctx.Queue()
        self.result_queue = ctx.Queue()
        self.pool = ctx.Pool(
        processes=dev_count,
        initializer=init_worker,
        initargs=(self.task_queue, self.result_queue),)
        

        for i in range(dev_count):
            self.pool.apply_async(
                generate_device,
                args=(model_path, i, False, batch_size),
            )

        
        
    def extract(self):
        return self.result_queue.get_nowait()
    
        
    def submit(self, tensor):
        self.task_queue.put(tensor)  

def init_worker(tasks, results):
        global task_queue, result_queue
        task_queue = tasks
        result_queue = results

def generate_device(model_path, device_index,is_half, batch_size):
        device_name = torch.device("cuda:" + str(device_index) if torch.cuda.is_available() else "cpu")
        resnet = ResNet()
        resnet.load_state_dict(torch.load(model_path))
        resnet.half() if is_half else resnet.float()
        resnet.to(device_name)
        
        while True:
            first = task_queue.get()
            if first is None:
                break
            
            inputs = [first]
            
            stop_after_batch = False
            
            while len(batch) < batch_size:
                try:
                    task = task_queue.get_nowait()
                except queue.Empty:
                    break

                if task is None:
                    stop_after_batch = True
                    break

                inputs.append(task)
            seeds, batch = map(list, zip(*inputs))
            batch = torch.stack(batch)
            batch = batch.to(device_name)
            with torch.no_grad():
                resnet.eval()
                outputs = resnet(batch)
                outputs = outputs.cpu()
            results = list(zip(seeds, outputs))
            for seed, output in results:
                result_queue.put((seed, output))
                
            if stop_after_batch:
                break