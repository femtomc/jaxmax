import jax
import jax.numpy as jnp

from juju import make_max_graph, jit


def fn(x):
    return jnp.sum(x)

print(jit(fn)(jnp.ones(10)).to_numpy())