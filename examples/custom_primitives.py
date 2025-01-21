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


def mandelbrot_max_lowering_rule(
    min_x,
    min_y,
    scale_x,
    scale_y,
    max_iterations,
    **params,
):
    width = params["width"]
    height = params["height"]
    output_dtype = DType.int32
    return ops.custom(
        name="mandelbrot",
        values=[min_x, min_y, scale_x, scale_y, max_iterations],
        out_types=[TensorType(dtype=output_dtype, shape=[height, width])],
    )[0].tensor


def mandelbrot_abstract_eval(*args, **params):
    height = params["height"]
    width = params["width"]
    return ShapedArray((height, width), jnp.int32)


mandelbrot = Primitive(
    "mandelbrot",
    mandelbrot_max_lowering_rule,
    mandelbrot_abstract_eval,
    multiple_results=False,
)

WIDTH = 15
HEIGHT = 15


def compute_mandelbrot(min_x, min_y, max_x, max_y, max_iterations):
    scale_x = (MAX_X - MIN_X) / WIDTH
    scale_y = (MAX_Y - MIN_Y) / HEIGHT
    return mandelbrot(
        min_x,
        min_y,
        scale_x,
        scale_y,
        max_iterations,
        width=WIDTH,
        height=HEIGHT,
    )


MAX_ITERATIONS = 500
MIN_X = -1.5
MAX_X = 0.7
MIN_Y = -1.12
MAX_Y = 1.12

print(make_max_graph(compute_mandelbrot)(MIN_X, MIN_Y, MAX_X, MAX_Y, MAX_ITERATIONS))

print(
    timing(jit(compute_mandelbrot))(
        MIN_X, MIN_Y, MAX_X, MAX_Y, MAX_ITERATIONS
    ).to_numpy()
)
