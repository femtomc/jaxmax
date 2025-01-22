import time

import jax.numpy as jnp

from juju import gpu_engine, jit


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print(f"{time2 - time1:.6f} s")
        return ret

    return wrap


@jit(engine=gpu_engine())
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)


timing(jax_code)(5, 10)
timing(jax_code)(5, 10)


@jit
def new_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)


timing(new_code)(5, 10)
timing(new_code)(5, 10)
