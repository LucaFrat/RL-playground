import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64, device=device)

x = torch.tensor(3.0, requires_grad=True)
w = torch.tensor(2.0)

y = x * w + 4

y.backward()

print(f'grad x: {x.grad}')
print(f'grad w: {w.grad}')