from max.dtype import DType
from max.graph import TensorType, ops

from juju import Primitive


def mandelbrot_rule(
    min_x,
    min_y,
    scale_x,
    scale_y,
    width,
    height,
    max_iterations,
):
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


mandelbrot = Primitive(
    "mandelbrot",
    mandelbrot_rule,
    mandelbrot_abstract_eval,
    multiple_results=False,
)


def compute_mandelbrot():
    v = mandelbrot()
    return v + 5.0


print(compute_mandelbrot())
