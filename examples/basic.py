import jax
import jax.numpy as jnp

from juju import jit


def fn(x):
    return jnp.reshape(x, (3, 3))


print(jax.make_jaxpr(fn)(jnp.ones(9)))

print(jit(fn)(jnp.ones(9)).to_numpy())
