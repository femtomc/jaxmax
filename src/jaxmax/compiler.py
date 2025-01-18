import functools
from dataclasses import dataclass, field

import beartype.typing as btyping
import jax.core as jc
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import util as jax_util
from jax.extend import linear_util as lu
from jax.interpreters import partial_eval as pe
from jax.util import safe_map
from max import engine
from max.driver import CPU
from max.graph import Graph, TensorType, TensorValue

from jaxmax.rules import max_rules, max_types

Any = btyping.Any
VarOrLiteral = jc.Var | jc.Literal
Callable = btyping.Callable
WrappedFunWithAux = tuple[lu.WrappedFun, Callable[[], Any]]


def get_shaped_aval(x):
    return jc.raise_to_shaped(jc.get_aval(x))


# The point of caching here is that, when JAX encounters a function that it needs to convert to a Jaxpr, if it has already done that before, save the work!
@lu.cache
def cached_stage_dynamic(flat_fun, in_avals):
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)
    typed_jaxpr = jc.ClosedJaxpr(jaxpr, consts)
    return typed_jaxpr


# The "graph capture" transformation is a "final style" one --
# it has a custom JAX Trace and Tracer type.
# This function is part of that style.
# We don't really use this style in our own transformation, we
# only use one of those transformations (cached_stage_dynamic)
# to get a Jaxpr.
@lu.transformation_with_aux
def _flatten_fun_nokwargs(in_tree, *args_flat):
    py_args = jtu.tree_unflatten(in_tree, args_flat)
    ans = yield py_args, {}
    yield jtu.tree_flatten(ans)


# Wrapper to assign a correct type.
flatten_fun_nokwargs: Callable[[lu.WrappedFun, Any], WrappedFunWithAux] = (
    _flatten_fun_nokwargs  # pyright: ignore[reportAssignmentType]
)


def stage(f):
    """Returns a function that stages a function to a ClosedJaxpr."""

    def wrapped(*args, **kwargs):
        fun = lu.wrap_init(f, kwargs)
        flat_args, in_tree = jtu.tree_flatten(args)
        flat_fun, out_tree = flatten_fun_nokwargs(fun, in_tree)
        flat_avals = safe_map(get_shaped_aval, flat_args)
        typed_jaxpr = cached_stage_dynamic(flat_fun, tuple(flat_avals))
        return typed_jaxpr, (flat_args, in_tree, out_tree)

    return wrapped


###################
# Our interpreter #
###################


@dataclass
class Environment:
    """Keeps track of variables and their values during interpretation."""

    env: dict[int, Any] = field(default_factory=dict)

    def read(self, var: VarOrLiteral) -> Any:
        if isinstance(var, jc.Literal):
            return var.val
        else:
            v = self.env.get(var.count)
            if v is None:
                raise ValueError(
                    f"Unbound variable in interpreter environment at count {var.count}:\nEnvironment keys (count): {list(self.env.keys())}"
                )
            return v

    def get(self, var: VarOrLiteral) -> Any:
        if isinstance(var, jc.Literal):
            return tensor_value(var.val)
        else:
            return self.env.get(var.count)

    def write(self, var: VarOrLiteral, cell: Any) -> Any:
        if isinstance(var, jc.Literal):
            return cell
        cur_cell = self.get(var)
        if isinstance(var, jc.DropVar):
            return cur_cell
        self.env[var.count] = cell
        return self.env[var.count]

    def __getitem__(self, var: VarOrLiteral) -> Any:
        return self.read(var)

    def __setitem__(self, key, val):
        raise ValueError(
            "Environments do not support __setitem__. Please use the "
            "`write` method instead."
        )

    def __contains__(self, var: VarOrLiteral):
        if isinstance(var, jc.Literal):
            return True
        return var.count in self.env

    def copy(self):
        keys = list(self.env.keys())
        return Environment({k: self.env[k] for k in keys})


def tensor_type(x):
    if isinstance(x, TensorType) or isinstance(x, TensorValue):
        return x
    x = jnp.array(x, copy=False)
    return TensorType(max_types[x.dtype], x.shape)


def tensor_value(x):
    return TensorValue(tensor_type(x), x)


@dataclass
class MAXInterpreter:
    def _eval_jaxpr_max(
        self,
        name: str,
        _jaxpr: jc.Jaxpr,
        consts: list[Any],
        args: list[Any],
    ):
        env = Environment()
        symbolic_args = map(tensor_type, args)
        symbolic_consts = map(tensor_value, consts)
        with Graph(name, input_types=list(symbolic_args)) as graph:
            jax_util.safe_map(env.write, _jaxpr.invars, graph.inputs)
            jax_util.safe_map(env.write, _jaxpr.constvars, symbolic_consts)
            for eqn in _jaxpr.eqns:
                invals = jax_util.safe_map(env.read, eqn.invars)
                subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                args = subfuns + invals
                rule = max_rules[eqn.primitive]
                outvals = rule(*args, **params)
                if not eqn.primitive.multiple_results:
                    outvals = [outvals]
                jax_util.safe_map(env.write, eqn.outvars, outvals)

            graph.output(*jax_util.safe_map(env.read, _jaxpr.outvars))

        return graph

    def run_interpreter(self, fn, *args, **kwargs):
        def _inner(*args):
            return fn(*args, **kwargs)

        _closed_jaxpr, (flat_args, _, out_tree) = stage(_inner)(*args)
        _jaxpr, consts = _closed_jaxpr.jaxpr, _closed_jaxpr.literals
        graph_out = self._eval_jaxpr_max(
            fn.__qualname__,
            _jaxpr,
            consts,
            flat_args,
        )
        return out_tree(), graph_out


def _max(f: Callable[..., Any]):
    @functools.wraps(f)
    def wrapped(*args):
        interpreter = MAXInterpreter()
        return interpreter.run_interpreter(f, *args)

    return wrapped


def max_graph(f: Callable[..., Any]):
    @functools.wraps(f)
    def wrapped(*args):
        _, graph = _max(f)(*args)
        return graph

    return wrapped


def max(
    f: Callable[..., Any],
    device=CPU(),
    path="./kernels.mojopkg",
):
    @functools.wraps(f)
    def wrapped(*args):
        defout, graph = _max(f)(*args)
        session = engine.InferenceSession(
            devices=[device],
            custom_extensions=path,
        )
        model = session.load(graph)
        ret = model.execute(*args)
        return jtu.tree_unflatten(defout, ret)

    return wrapped
