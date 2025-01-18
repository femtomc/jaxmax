from dataclasses import dataclass, field
from jaxmax.compiler import _max
import jax.tree_util as jtu
from max import engine

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
    f: callable

    def __call__(self, *args):
        jit_key = jtu.tree_structure(args)
        if (self.f, jit_key) in global_jit_engine.cache:
            compiled, _ = global_jit_engine.cache[(self.f, jit_key)]
            return compiled(*args)
        else:
            defout, graph = _max(self.f)(*args)
            model = global_jit_engine.load(graph)
            def _compiled(*args):
                return jtu.tree_unflatten(defout, model.execute(*args))
            global_jit_engine[(self.f, jit_key)] = (_compiled, graph)
            return _compiled(*args)

def jit(f: callable):
    return JITFunction(f)