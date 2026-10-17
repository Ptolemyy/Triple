from model import ResNet
import torch

resnet = ResNet()

device = torch.device("cuda:0")
input = torch.rand(50000,8,6,6)
batch = torch.stack([i for i in input])
batch = batch.to(device)
batch = batch.half()
resnet = resnet.to(device)
resnet = resnet.half()
with torch.no_grad():
    output = resnet(batch)

output = output.cpu()
print(output)