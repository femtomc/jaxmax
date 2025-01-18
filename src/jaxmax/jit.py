from dataclasses import dataclass, field

import jax.tree_util as jtu
from beartype.typing import Callable
from max import engine

from jaxmax.compiler import _max


@dataclass
class JITEngine:
    cache: dict = field(default_factory=dict)
    session = engine.InferenceSession()

    def load(self, graph):
        return self.session.load(graph)

    def __getitem__(self, key):
        return self.cache[key]

    def __hasitem__(self, key):
        return key in self.cache

    def __setitem__(self, key, val):
        self.cache[key] = val


global_jit_engine = JITEngine()


@dataclass
class JITFunction:
    f: Callable[..., any]

    def __call__(self, *args):
        jit_key = jtu.tree_structure(args)
        if (self.f, jit_key) in global_jit_engine.cache:
            compiled, _ = global_jit_engine.cache[self.f, jit_key]
            return compiled(*args)
        else:
            # Static tracing to generate a graph.
            defout, graph = _max(self.f)(*args)
            model = global_jit_engine.load(graph)

            # Generate a function which executes the graph using
            # the model stored in the session.
            def _compiled(*args):
                flat_args = jtu.tree_leaves(args)
                return jtu.tree_unflatten(defout, model.execute(*flat_args))

            # Store the function.
            global_jit_engine[self.f, jit_key] = (_compiled, graph)
            return _compiled(*args)


def jit(f: Callable[..., any]):
    return JITFunction(f)
