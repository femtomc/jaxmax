import jax.numpy as jnp
from jax import grad

from juju import max, max_graph


@max
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)


print(jax_code(5, 10))


@max_graph
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)


print(jax_code(5, 10))


@max
@grad
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)


print(jax_code(5.0, 10.0))
