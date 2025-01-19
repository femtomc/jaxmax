from dataclasses import dataclass, field

import beartype.typing as btyping
import numpy as np
from jax._src import ad_util, prng
from jax.extend.core import Primitive, primitives
from max.dtype import DType
from max.graph import TensorType, ops

Callable = btyping.Callable

max_types = {
    np.dtype(np.uint32): DType.uint32,
    np.dtype(np.int32): DType.int32,
    np.dtype(np.float32): DType.float32,
}


@dataclass
class Ruleset:
    max_rules: dict[Primitive, Callable[[TensorType, ...], TensorType]] = field(
        default_factory=dict
    )

    def register(self, jax_primitive: Primitive, max_primitive):
        self.max_rules[jax_primitive] = max_primitive

    def register_def(self, jax_primitive: Primitive):
        def _register(rule):
            self.max_rules[jax_primitive] = rule

        return _register

    def __getitem__(self, jax_primitive: Primitive):
        return self.max_rules[jax_primitive]


max_rules = Ruleset()

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


@max_rules.register_def(primitives.div_p)
def div(x, y, **params):
    return ops.div(x, y)


@max_rules.register_def(primitives.integer_pow_p)
def integer_pow(x, y, **params):
    return ops.pow(x, y)


@max_rules.register_def(primitives.reduce_sum_p)
def reduce_sum(x, **params):
    ret = ops.custom(
        name="reduce_sum",
        values=[x],
        out_types=[TensorType(dtype=x.dtype, shape=x.tensor.shape)],
    )
    return ret[0]


@max_rules.register_def(primitives.broadcast_in_dim_p)
def broadcast_in_dim(x, **params):
    assert False, "broadcast_in_dim not implemented"


@max_rules.register_def(primitives.neg_p)
def neg(x, **params):
    return ops.mul(x, -1)


@max_rules.register_def(ad_util.add_any_p)
def add_any(x, y, **params):
    return ops.add(x, y)


@max_rules.register_def(primitives.convert_element_type_p)
def convert_element_type(x, **params):
    return ops.cast(x, dtype=max_types[params["new_dtype"]])


##############
# Randomness #
##############


# These are primitives which JAX may eventually deprecate,
# and deal with conversion from custom key types to uint32 and back.
@max_rules.register_def(prng.random_wrap_p)
def random_wrap(x, **params):
    return x


@max_rules.register_def(prng.random_unwrap_p)
def random_unwrap(x, **params):
    return x


@max_rules.register_def(prng.random_split_p)
def random_split(x, **params):
    ret = ops.custom(
        name="random_split",
        values=[x],
        out_types=[TensorType(dtype=x.dtype, shape=x.tensor.shape)],
    )
    return ret[0]
