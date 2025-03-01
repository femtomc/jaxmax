from dataclasses import dataclass, field

import beartype.typing as btyping
import numpy as np
from jax._src import ad_util
from jax.extend.core import Primitive, primitives
from max.dtype import DType
from max.graph import TensorType, ops

max_types = {
    np.dtype(np.uint32): DType.uint32,
    np.dtype(np.int32): DType.int32,
    np.dtype(np.float32): DType.float32,
}
"""
Conversion from NumPy dtypes to MAX dtypes.
"""


@dataclass
class Ruleset:
    rules: dict[Primitive, btyping.Callable[[TensorType, ...], TensorType]] = field(
        default_factory=dict
    )

    def register(self, jax_primitive: Primitive, max_primitive):
        assert jax_primitive not in self.rules, jax_primitive
        self.rules[jax_primitive] = max_primitive

    def register_def(self, jax_primitive: Primitive):
        def _register(rule):
            assert jax_primitive not in self.rules, jax_primitive
            self.rules[jax_primitive] = rule

        return _register

    def __getitem__(self, jax_primitive: Primitive):
        return self.rules[jax_primitive]

    def keys(self):
        return self.rules.keys()


max_rules = Ruleset()
"""
Global rule dictionary used by the lowering interpreter.
"""


####################
# Registered rules #
####################

max_rules.register(primitives.add_p, ops.add)
max_rules.register(primitives.mul_p, ops.mul)
max_rules.register(primitives.sub_p, ops.sub)
max_rules.register(primitives.sin_p, ops.sin)
max_rules.register(primitives.cos_p, ops.cos)
max_rules.register(primitives.abs_p, ops.abs)
max_rules.register(primitives.max_p, ops.max)
max_rules.register(primitives.min_p, ops.min)
max_rules.register(primitives.exp_p, ops.exp)
max_rules.register(primitives.log_p, ops.log)
max_rules.register(primitives.floor_p, ops.floor)


@max_rules.register_def(primitives.acos_p)
def acos(x, **params):
    ret = ops.custom(
        name="acos",
        values=[x],
        out_types=[TensorType(dtype=x.dtype, shape=x.tensor.shape)],
    )
    return ret[0]


# This is dumb -- fix with kernel iota later.
@max_rules.register_def(primitives.iota_p)
def iota(**params):
    shape = params["shape"]
    dtype = params["dtype"]
    return ops.constant(
        np.arange(0, shape[0], dtype=dtype),
        dtype=max_types[dtype],
    )


@max_rules.register_def(primitives.div_p)
def div(x, y, **params):
    return ops.div(x, y)


@max_rules.register_def(primitives.integer_pow_p)
def integer_pow(x, y, **params):
    return ops.pow(x, y)


@max_rules.register_def(primitives.reduce_sum_p)
def reduce_sum(x, **params):
    axes = params["axes"]
    assert len(axes) == 1, "Only a single axis is currently allowed."
    (axis,) = axes
    return ops.sum(x, axis=axis)


@max_rules.register_def(primitives.neg_p)
def neg(x, **params):
    return ops.mul(x, -1)


@max_rules.register_def(ad_util.add_any_p)
def add_any(x, y, **params):
    return ops.add(x, y)


@max_rules.register_def(primitives.convert_element_type_p)
def convert_element_type(x, **params):
    return ops.cast(x, dtype=max_types[params["new_dtype"]])


@max_rules.register_def(primitives.reshape_p)
def reshape(x, **params):
    return ops.reshape(x, params["new_sizes"])


@max_rules.register_def(primitives.broadcast_in_dim_p)
def broadcast_in_dim(x, **params):
    shape = params["shape"]
    return ops.broadcast_to(x, shape)


@max_rules.register_def(primitives.concatenate_p)
def concatenate(*args, **params):
    return ops.concat(list(args), axis=params["dimension"])
