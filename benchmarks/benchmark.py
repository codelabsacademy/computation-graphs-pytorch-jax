"""
Micro-benchmark comparing PyTorch eager, torch.compile, and JAX jit
for a single linear layer at two batch sizes.
"""

import time, os, warnings
import numpy as np

import torch
import jax, jax.numpy as jnp


warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


torch.manual_seed(0)
key = jax.random.PRNGKey(0)

BATCHES = [32, 2048]
D_IN, D_OUT = 1024, 1024
STEPS = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = os.environ.get("DEVICE", DEVICE)

# PyTorch model
pytorch_linear = torch.nn.Linear(D_IN, D_OUT).to(DEVICE)

# Compiled PyTorch
compiled_linear = torch.compile(pytorch_linear)

# JAX model
w = jax.random.normal(key, (D_IN, D_OUT))
jax_linear = jax.jit(lambda x, w: x @ w)

def bench(fn, batch):
    t0 = time.perf_counter()
    for _ in range(STEPS):
        _ = fn(batch)
    return (time.perf_counter() - t0) / STEPS * 1000  # ms

for bs in BATCHES:
    x_pt = torch.randn(bs, D_IN, device=DEVICE)
    x_jax = jax.random.normal(key, (bs, D_IN))

    pt = bench(lambda b: pytorch_linear(b).cpu(), x_pt)
    ptc = bench(lambda b: compiled_linear(b).cpu(), x_pt)
    jx = bench(lambda b: jax_linear(b, w).block_until_ready(), x_jax)

    print(f"Batch {bs:5d} | PyTorch: {pt:5.2f} ms | torch.compile: {ptc:5.2f} ms | JAX: {jx:5.2f} ms")
