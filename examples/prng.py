from jax import random as jrand

from juju import jit


def fn(key):
    return jrand.split(key)


print(jit(fn)(jrand.PRNGKey(1)))
