from dataclasses import dataclass, field

import beartype.typing as btyping
import jax.core as jc
from jax import lax
from jax._src import ad_util, prng
from max.graph import Graph, TensorType, ops

Callable = btyping.Callable


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