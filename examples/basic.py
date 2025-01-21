import jax
import jax.numpy as jnp

from juju import max_execute


@max_execute
def fn(x):
    return jax.lax.iota(float, 10)


print(fn(jnp.array(0.7)))

print(jax.lax.iota(float, 10))
