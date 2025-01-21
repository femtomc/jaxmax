import time

import jax.numpy as jnp
from jax.core import ShapedArray
from max.dtype import DType
from max.graph import TensorType, ops

from juju import Primitive, jit, make_max_graph


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print(f"{time2 - time1:.6f} s")
        return ret

    return wrap


def mandelbrot_max_lowering_rule(**params):
    min_x = params["min_x"]
    min_y = params["min_y"]
    scale_x = params["scale_x"]
    scale_y = params["scale_y"]
    width = params["width"]
    height = params["height"]
    max_iterations = params["max_iterations"]
    output_dtype = DType.int32
    return ops.custom(
        name="mandelbrot",
        values=[
            ops.constant(min_x, dtype=DType.float32),
            ops.constant(min_y, dtype=DType.float32),
            ops.constant(scale_x, dtype=DType.float32),
            ops.constant(scale_y, dtype=DType.float32),
            ops.constant(max_iterations, dtype=DType.int32),
        ],
        out_types=[TensorType(dtype=output_dtype, shape=[height, width])],
    )[0].tensor


def mandelbrot_abstract_eval(**params):
    height = params["height"]
    width = params["width"]
    return ShapedArray((height, width), jnp.int32)


mandelbrot = Primitive(
    "mandelbrot",
    mandelbrot_max_lowering_rule,
    mandelbrot_abstract_eval,
    multiple_results=False,
)


def compute_mandelbrot():
    WIDTH = 15
    HEIGHT = 15
    MAX_ITERATIONS = 100
    MIN_X = -1.5
    MAX_X = 0.7
    MIN_Y = -1.12
    MAX_Y = 1.12
    scale_x = (MAX_X - MIN_X) / WIDTH
    scale_y = (MAX_Y - MIN_Y) / HEIGHT
    return mandelbrot(
        min_x=MIN_X,
        min_y=MIN_Y,
        scale_x=scale_x,
        scale_y=scale_y,
        width=WIDTH,
        height=HEIGHT,
        max_iterations=MAX_ITERATIONS,
    )


print(make_max_graph(compute_mandelbrot)())
print(jit(compute_mandelbrot)().to_numpy())
timing(jit(compute_mandelbrot))()
