# import torch
# import torchvision
# from torch.utils import data
# from torchvision import transforms
# from d2l import torch as d2l

# trans = transforms.ToTensor()
# train_data = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=trans)
# test_data = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=trans)

# # train_iter = data.DataLoader(train_data, batch_size=64, shuffle=True)
# # test_iter = data.DataLoader(test_data, batch_size=64, shuffle=True)

# def get_fashion_mnist_labels(labels):
#     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     return [text_labels[int(i)] for i in labels]

# # def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
# #     figsize = (num_cols * scale, num_rows * scale)
# #     _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
# #     axes = axes.flatten()
# #     for i, (ax, img) in enumerate(zip(axes, imgs)):
# #         ax.imshow(img.numpy())
# #         ax.axes.get_xaxis().set_visible(False)
# #         ax.axes.get_yaxis().set_visible(False)
# #     return axes.

# batch_size = 256
# def get_dataloader_workers():
#     return 4

# train_iter = data.DataLoader(train_data, batch_size, shuffle=True, num_workers=get_dataloader_workers())
# timer = d2l.Timer()
# for X, y in train_iter:
#     continue
# print(f'{timer.stop():.2f} sec')

# def load_data_fashion_mnist(batch_size, resize=None):
#     trans = [transforms.ToTensor()]
#     if resize:
#         trans.insert(0, transforms.Resize(resize))
#     trans = transforms.Compose(trans)
#     mnist_train = torchvision.datasets.FashionMNIST(root='../data', train=True, download=True, transform=trans)
#     mnist_test = torchvision.datasets.FashionMNIST(root='../data', train=False, download=True, transform=trans)
#     return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
#             data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=get_dataloader_workers()))

# train_iter, test_iter = load_data_fashion_mnist(batch_size)
# for X, y in train_iter:
#     print(X.shape, X.dtype, y.shape, y.dtype)
#     break

# num_inputs = 784
# num_outputs = 10

# w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# b = torch.zeros(num_outputs, requires_grad=True)
# X = torch.randn(2, 784)  # 创建一个2个样本，每个样本784维的测试输入
# y = torch.tensor([0, 2])

# def softmax(X):
#     X_exp = torch.exp(X)
#     partition = X_exp.sum(1, keepdim=True)
#     return X_exp / partition

# X = torch.normal(0, 1, (2, 5))
# X_prob = softmax(X)
# print(X_prob, X_prob.sum(1))

# def net(X):
#     return softmax(torch.matmul(X.reshape((-1, w.shape[0])), w) + b)

# # 测试网络输出
# print(net(X)[0:10])  # 打印前10个预测结果
# print(net(X)[:10].argmax(1))
# print(net(X)[:10].argmax(1) == y)

# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y = torch.tensor([0, 2])
# print(d2l.accuracy(y_hat, y))
# def cross_entropy(y_hat, y):
#     return -torch.log(y_hat[range(len(y_hat)), y])

# cross_entropy(y_hat, y)

# def accuracy(y_hat, y):
#     if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
#         y_hat = y_hat.argmax(axis=1)
#     cmp = y_hat.type(y.dtype) == y
#     return float(cmp.type(y.dtype).sum()) / len(cmp)

# def evaluate_accuracy(net, data_iter):
#     if isinstance(net, torch.nn.Module):
#         net.eval()
#     metric = d2l.Accumulator(2)
#     with torch.no_grad():
#         for X, y in data_iter:
#             metric.add(accuracy(net(X), y), y.numel())
#     return metric[0] / metric[1]

# class Accumulator:
#     def __init__(self, n):
#         self.data = [0.0] * n

#     def add(self, *args):
#         self.data = [a + float(b) for a, b in zip(self.data, args)]

#     def reset(self):
#         self.data = [0.0] * len(self.data)
    
#     def __getitem__(self, idx):
#         return self.data[idx]
    
# evaluate_accuracy(net, test_iter)

# if __name__ == '__main__':
#     batch_size = 256
#     train_iter, test_iter = load_data_fashion_mnist(batch_size)
    
#     timer = d2l.Timer()
#     for X, y in train_iter:
#         continue
#     print(f'{timer.stop():.2f} sec')
    
#     # 测试数据形状
#     for X, y in train_iter:
#         print(X.shape, X.dtype, y.shape, y.dtype)
#         break
    
#     # 测试 softmax
#     X_test = torch.normal(0, 1, (2, 5))
#     X_prob = softmax(X_test)
#     print(X_prob, X_prob.sum(1))
    
#     # 测试网络
#     X_test = torch.randn(2, 784)  # 创建正确大小的测试输入
#     y_test = torch.tensor([0, 2])
    
#     # 使用 X_test 替代 X
#     print(net(X_test)[:10])  # 打印前10个预测结果
#     print(net(X_test)[:10].argmax(1))