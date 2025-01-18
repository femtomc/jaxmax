import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

from jaxmax import max


# This is a convenience function designed to support equality
# comparison between executing a function using JAX's JIT
# and executing a function using MX compile.
def jax_equality_assertion(fn, *args):
    def check(v):
        if not isinstance(v, bool) and v.shape:
            return all(v)
        else:
            return v

    assert check(pytest.approx(jax.jit(fn)(*args), 1e-5) == max(fn)(*args))


def tire_kick_assertion(fn, *args):
    assert jnp.any(max(fn)(*args))


class TestCompiler:
    def test_add_p(self):
        jax_equality_assertion(lambda x, y: x + y, 5.0, 5.0)
