"""
Graph-inspection helpers for PyTorch and JAX.
Compatible with Python 3.8 +  🐍
"""

# ------------------  PyTorch  ------------------ #
import torch
from typing import Set

def dump_graph(node, visited: Set[torch.autograd.Function] = None, depth: int = 0):
    """
    Recursively print the autograd graph of a PyTorch Tensor **or**
    an autograd Function node, without assigning to read-only attrs.

    Example
    -------
    >>> y = model(x).sum()
    >>> dump_graph(y)
    """
    if visited is None:
        visited = set()

    # Accept either a Tensor or a Function
    grad_fn = node.grad_fn if isinstance(node, torch.Tensor) else node
    if grad_fn is None or grad_fn in visited:
        return

    visited.add(grad_fn)
    print("  " * depth + repr(grad_fn))

    for next_fn, _ in grad_fn.next_functions:
        if next_fn is not None:
            dump_graph(next_fn, visited, depth + 1)

# ------------------  JAX  ------------------ #
def count_ops(f, *example_args, **example_kwargs):
    """
    Return the number of primitive operations in a JAX function.
    """
    import jax
    jaxpr = jax.make_jaxpr(f)(*example_args, **example_kwargs)
    return len(jaxpr.jaxpr.eqns)