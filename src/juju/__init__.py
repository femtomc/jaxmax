from beartype import BeartypeConf
from beartype.claw import beartype_this_package

conf = BeartypeConf(
    is_color=True,
    is_debug=False,
    is_pep484_tower=True,
    violation_type=TypeError,
)

beartype_this_package(conf=conf)

from .compiler import jit, make_max_graph
from .primitive import Primitive

__all__ = [
    "Primitive",
    "jit",
    "make_max_graph",
]
