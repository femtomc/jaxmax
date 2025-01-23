# Lowering rules

Here, we list the set of JAX primitives with MAX lowering rules. These rules are used internally by `juju`'s lowering interpreter, and this list reflects the coverage of lowering over JAX's primitives.

```python exec="on" source="material-block"
from juju.rules import max_rules
for primitive in list(max_rules.max_rules.keys()):
    print(primitive)
```

The implementation of these rules can be found in the [`juju.rules`](https://github.com/femtomc/juju/blob/main/src/juju/rules.py) module.