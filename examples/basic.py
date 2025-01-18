from jaxmax import max
import jax.numpy as jnp


@max
def fn(x, y):
    return x + y + 1


print(fn(jnp.array(5), jnp.array(6)))
