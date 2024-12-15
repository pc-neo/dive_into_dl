import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
# print(net(X))

# print(net[2].state_dict())

print(net[2].bias)
print(net[2].bias.data)
print(net[2].bias.grad)
# print(net[2].weight.data)
print(*[(name, param.shape) for name, param in net.named_parameters()])
# print(net[2].weight.grad)
print(net)

def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 1))

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(1, 1))
print(rgnet)

rgnet[0][1].weight.data[0] += 1