import time

import jax.numpy as jnp
import jax

from juju import gpu_engine, jit


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print(f"{time2 - time1:.6f} s")
        return ret

    return wrap

def fn(x):
    for i in range(600000):
        x = x + 1
    return x

print(len(jax.make_jaxpr(fn)(1.0).eqns))
timing(jax.jit(fn).lower)(1.0)