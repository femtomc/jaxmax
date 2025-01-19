from juju import jit
from jax import grad, vmap
import jax.numpy as jnp


def f(x):
    return x**2


loss = jit(grad(lambda x: jnp.sum(vmap(f)(x))))

print(loss(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])))
