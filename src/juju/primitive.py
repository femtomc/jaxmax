from beartype.typing import Callable
from jax.extend.core import Primitive as JPrim

from juju.rules import max_rules


def Primitive(
    name: str,
    max_lowering_rule: Callable,
    jax_abstract_evaluation_rule: Callable,
    multiple_results=True,
):
    """
    Construct a new JAX primitive, and register `jax_abstract_evaluation_rule`
    as the abstract evaluation rule for the primitive for JAX, and `max_lowering_rule` for `juju`'s lowering interpreter.

    Returns a function that invokes the primitive via JAX's `Primitive.bind` method.
    """
    new_prim = JPrim(name + "_p")
    new_prim.def_abstract_eval(jax_abstract_evaluation_rule)
    max_rules.register(new_prim, max_lowering_rule)

    # JAX can't execute the code by itself!
    # We have to use MAX, so we raise an exception if JAX tries to evaluate the primitive.
    def _raise_impl(*args, **params):
        raise Exception(f"{name} is a MAX primitive, cannot be evaluated by JAX.")

    new_prim.def_impl(_raise_impl)

    def _invoke(*args, **params):
        return new_prim.bind(*args, **params)

    return _invoke
