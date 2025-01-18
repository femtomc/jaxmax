from dataclasses import dataclass, field

import beartype.typing as btyping
import jax.core as jc
import numpy as np
from jax import lax
from max.dtype import DType
from max.graph import TensorType, ops

Callable = btyping.Callable

max_types = {
    np.dtype(np.int32): DType.int32,
    np.dtype(np.float32): DType.float32,
}


@dataclass
class Ruleset:
    max_rules: dict[jc.Primitive, Callable[[TensorType, ...], TensorType]] = field(
        default_factory=dict
    )

    def register(self, jax_primitive: jc.Primitive, max_primitive):
        self.max_rules[jax_primitive] = max_primitive

    def register_def(self, jax_primitive: jc.Primitive):
        def _register(rule):
            self.max_rules[jax_primitive] = rule

        return _register

    def __getitem__(self, jax_primitive: jc.Primitive):
        return self.max_rules[jax_primitive]


max_rules = Ruleset()

####################
# Registered rules #
####################

max_rules.register(lax.add_p, ops.add)
max_rules.register(lax.mul_p, ops.mul)
max_rules.register(lax.sub_p, ops.sub)
max_rules.register(lax.sin_p, ops.sin)
max_rules.register(lax.abs_p, ops.abs)


@max_rules.register_def(lax.convert_element_type_p)
def convert_element_type(x, **params):
    return ops.cast(x, dtype=max_types[params["new_dtype"]])