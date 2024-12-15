import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
# net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
# print(net[0].weight)

# X = torch.rand(2, 4)
# print(net(X))

class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
    
layer = CenteredLayer()
print(layer(torch.rand(4, 4)))