# import numpy as np
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor
# from d2l import torch as d2l

# batch_size = 256
# train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# num_inputs = 784
# num_outputs = 10
# num_hiddens = 256

# w1 = torch.normal(0, 0.01, size=(num_inputs, num_hiddens), requires_grad=True)
# b1 = torch.zeros(num_hiddens, requires_grad=True)
# w2 = torch.normal(0, 0.01, size=(num_hiddens, num_outputs), requires_grad=True)
# b2 = torch.zeros(num_outputs, requires_grad=True)


# params = [w1, b1, w2, b2]

# def relu(x):
#     return torch.max(torch.tensor(0), x)

# def net(x):
#     x = x @ w1 + b1
#     x = relu(x)
#     x = x @ w2 + b2
#     return x

# def loss(y_hat, y):
#     return d2l.cross_entropy(y_hat, y)

# lr = 0.1
# num_epochs = 10
# optimizer = torch.optim.SGD(params, lr=lr)
# # d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

# for epoch in range(num_epochs):
#     for X, y in train_iter:
#         optimizer.zero_grad()
#         y_hat = net(X)
#         l = loss(y_hat, y)
#         l.backward()
#         optimizer.step()
#     # print(f'epoch {epoch + 1}, loss {l:f}')
