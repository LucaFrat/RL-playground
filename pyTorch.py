import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64, device=device)

class NeuralNetwork(nn.Module):
    def __init__(self, in_dim=784, out_dim=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.seq = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.seq(x)
        return logits

model = NeuralNetwork(in_dim=28*28, out_dim=10)
print(model)
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")