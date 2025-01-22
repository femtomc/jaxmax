import jax
import jax.numpy as jnp

from juju import make_max_graph


def fn(x):
    return jnp.mean(x * jnp.linspace(x, 1, 50))


print(jax.make_jaxpr(fn)(-1.0))

print(make_max_graph(fn)(-1.0))
