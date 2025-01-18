import jax.numpy as jnp

from jaxmax import max, max_graph


@max
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)


print(jax_code(5, 10).to_numpy())


@max_graph
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)


print(jax_code(5, 10))
