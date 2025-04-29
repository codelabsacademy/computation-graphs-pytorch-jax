"""
Minimal demo: build a small network in JAX,
trace with make_jaxpr, and count ops.
"""

import jax, jax.numpy as jnp
from utils.graph_utils import count_ops

key = jax.random.PRNGKey(0)

def init_dense(in_dim, out_dim, key):
    k1, _ = jax.random.split(key)
    w = jax.random.normal(k1, (in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
    b = jnp.zeros((out_dim,))
    return w, b

w1, b1 = init_dense(4, 8, key)
w2, b2 = init_dense(8, 2, key)

def mlp(x, w1, b1, w2, b2):
    hidden = jax.nn.relu(x @ w1 + b1)
    return hidden @ w2 + b2

jit_mlp = jax.jit(mlp)

x = jnp.ones((1, 4))
out = jit_mlp(x, w1, b1, w2, b2)
print("Output:", out)

print("\n--- JAXPR ---")
print(jax.make_jaxpr(jit_mlp)(x, w1, b1, w2, b2))

print("Op count:", count_ops(mlp, x, w1, b1, w2, b2))
