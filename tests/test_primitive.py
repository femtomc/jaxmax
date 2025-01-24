import jax
import jax.numpy as jnp
import pytest
from jax import grad

from juju import jit, Primitive

from juju import Primitive, jit
from jax.core import ShapedArray
import jax.numpy as jnp
from max.graph import ops, TensorType

# Lowering rule to MAX, gets called by 
# juju's lowering interpreter.
def add_one_lowering(x, **params):
    return ops.custom(
        name="add_one", # needs to match your @compiler.register name
        values=[x],
        out_types=[TensorType(dtype=x.dtype, shape=x.tensor.shape)],
    )[0]

# Abstract evaluation rule for JAX, gets called
# by JAX when tracing a program to a Jaxpr.
def add_one_abstract(x, **params):
    return ShapedArray(x.shape, x.dtype)

# Register and coordinate everything, get a callable back.
add_one = Primitive(
    "add_one", # can be anything
    add_one_lowering, 
    add_one_abstract,
)

class TestPrimitive:
    def test_add_one(self):
        @jit(coerces_to_jnp=True)
        def jaxable_program(x):
            x = x * 2
            return add_one(x) 
        
        assert jnp.all(jaxable_program(jnp.ones(10)) == (jnp.ones(10) * 2 + 1))