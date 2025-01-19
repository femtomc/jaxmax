import jax.numpy as jnp
from jax import grad

from juju import make_max_graph, max_execute


@max_execute
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)


print(jax_code(5, 10))


@make_max_graph
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)


print(jax_code(5, 10))


@max_execute
@grad
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)


print(jax_code(5.0, 10.0))
