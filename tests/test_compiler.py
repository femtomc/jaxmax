import jax
import jax.numpy as jnp
import pytest
from jax import grad

from juju import max_execute


# This is a convenience function designed to support equality
# comparison between executing a function using JAX's JIT
# and executing a function using MX compile.
def jax_equality_assertion(fn, *args):
    def check(v):
        if not isinstance(v, bool) and v.shape:
            return all(v)
        else:
            return v

    assert check(pytest.approx(jax.jit(fn)(*args), 1e-5) == max_execute(fn)(*args))


def tire_kick_assertion(fn, *args):
    assert jnp.any(max_execute(fn)(*args))


class TestCompiler:
    def test_add_p(self):
        jax_equality_assertion(lambda x, y: x + y, 5.0, 5.0)

    def test_neg_p(self):
        jax_equality_assertion(lambda x: -x, 5.0)

    def test_grad(self):
        @grad
        def jax_code(x, y):
            v = x + y
            v = v * v
            return jnp.sin(v)

        jax_equality_assertion(jax_code, 5.0, 10.0)
