import torch
from torch import nn

# print(torch.device('cpu'))
# print(torch.device('cuda'))
# print(torch.device('cuda:1'))

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_name(1))
print(torch.cuda.get_device_name(2))
print(torch.cuda.get_device_name(3))

x = torch.ones(2, 3, device='cuda:0')
print(x)

y = torch.ones(2, 3, device='cuda:1')
print(y)

z = x.cuda(1)
print(z)

print(x + z)


net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device='cuda:1')
print(net)

