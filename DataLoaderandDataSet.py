import torch


x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x * 2
z = y.sum()
z.backward()

print(x.grad)
print(x)
print(y)
print(z)