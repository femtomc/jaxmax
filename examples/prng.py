from jax import random as jrand

from juju import max_execute


def fn(key):
    return jrand.split(key)


print(max_execute(fn)(jrand.PRNGKey(1)))
