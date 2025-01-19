import jax.numpy as jnp

from juju import max_execute


@max_execute
def fn(x, y):
    return x + y + 1


print(fn(jnp.array(5), jnp.array(6)))
