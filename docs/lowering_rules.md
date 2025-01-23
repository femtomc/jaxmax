# Lowering rules

Here, we list the set of JAX primitives with MAX lowering rules. These rules are used internally by `juju`'s lowering interpreter, and this list reflects the coverage of lowering over JAX's primitives.

Note that, even if a JAX primitive is in the list below, it's possible that _our semantics_ are incorrect or missing some configuration that JAX supports. Our test suite is the place where we test fidelity of the lowering, so if something appears to be misbehaving, please file an issue. If you're using `juju` and we're missing a lowering rule, also please file an issue!

```python exec="on" source="material-block"
from juju.rules import max_rules

for primitive in list(max_rules.max_rules.keys()):
    print(primitive)
```

The implementation of these rules can be found in the [`juju.rules`](https://github.com/femtomc/juju/blob/main/src/juju/rules.py) module.