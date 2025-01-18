from jax import make_jaxpr
from jax import random as jrand


def fn(key):
    return jrand.split(key)


print(make_jaxpr(fn)(jrand.PRNGKey(1)))
