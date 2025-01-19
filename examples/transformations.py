from juju import jit
from jax import grad, vmap, make_jaxpr
import jax.numpy as jnp


def f(x):
    return x**2


print(jit(vmap(f))(jnp.array([1.0, 2.0, 3.0])))
