"""
Minimal demo: build a small network in PyTorch,
visualise grad_fn chain, and dump the autograd graph.
"""

import torch
from utils.graph_utils import dump_graph

torch.manual_seed(0)

class SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim=4, hidden=8, out_dim=2):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

net = SimpleMLP()
x = torch.randn(1, 4, requires_grad=True)
y = net(x).sum()
print("Output:", y.item())
print("grad_fn chain head:", y.grad_fn)

print("\n--- Autograd Graph ---")
dump_graph(y)

y.backward()
print("Gradient on x:", x.grad)
