from jax import random as jrand

from juju import max


def fn(key):
    return jrand.split(key)


print(max(fn)(jrand.PRNGKey(1)))