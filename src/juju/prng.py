from jax._src import prng
from max.graph import TensorType, ops

from juju.rules import max_rules

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
