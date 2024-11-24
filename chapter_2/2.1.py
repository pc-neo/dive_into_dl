import torch

print(torch.__version__)

a = torch.arange(6, dtype=torch.float32).reshape(3, 2)
print(f"before: {id(a)  }")
b = torch.arange(4, dtype=torch.float32).reshape(2, 2)
print(f"before: {id(b)}")
print(a)
print(b)
print(torch.matmul(a, b))

# 直接赋值，会分配新内存
# 使用[:]赋值，不会分配新内存
c = torch.matmul(a, b)
print(f"after: {id(a)}")


print(a < c)
