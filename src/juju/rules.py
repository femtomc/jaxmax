from dataclasses import dataclass, field

import beartype.typing as btyping
import numpy as np
from jax import lax
from jax._src import ad_util, prng
from jax.extend.core import Primitive
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

max_rules.register(lax.add_p, ops.add)
max_rules.register(lax.mul_p, ops.mul)
max_rules.register(lax.sub_p, ops.sub)
max_rules.register(lax.sin_p, ops.sin)
max_rules.register(lax.cos_p, ops.cos)
max_rules.register(lax.abs_p, ops.abs)


@max_rules.register_def(lax.neg_p)
def neg(x, **params):
    return ops.mul(x, -1)


@max_rules.register_def(ad_util.add_any_p)
def add_any(x, y, **params):
    return ops.add(x, y)


@max_rules.register_def(lax.convert_element_type_p)
def convert_element_type(x, **params):
    return ops.cast(x, dtype=max_types[params["new_dtype"]])


##############
# Randomness #
##############


@max_rules.register_def(prng.random_wrap_p)
def random_wrap(x, **params):
    return x


@max_rules.register_def(prng.random_split_p)
def random_split(x, **params):
    ret = ops.custom(
        name="random_split",
        values=[x],
        out_types=[TensorType(dtype=x.dtype, shape=x.tensor.shape)],
    )
    return ret[0]


@max_rules.register_def(prng.random_unwrap_p)
def random_unwrap(x, **params):
    return x
