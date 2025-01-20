import time

from max import engine

import juju


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print(f"{time2 - time1:.6f} s")
        return ret

    return wrap


def foo(x, y):
    v = x + y
    for i in range(10000):
        v = v + 1
    return v


graph = timing(juju.make_max_graph)(foo)(5, 10)
session = engine.InferenceSession(num_threads=8)
model = timing(session.load)(graph)
retval = timing(model.execute)(5, 10)[0]
print(retval.to_numpy())
