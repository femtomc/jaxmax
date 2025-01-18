# jaxmax

> [!CAUTION]
> This package is a rather simple and dumb idea, which probably has some sharp edges. Simple programs only for now!

This package supports a API called `max` whose purpose is to transform a _JAX computation_ into a [MAX computation graph](https://docs.modular.com/max). These graphs can then be executed using MAX.

**Example:**
```python
import jax.numpy as jnp
from jaxmax import max

@max
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)

print(jax_code(5, 10).to_numpy()) # -0.93009484
```

The way this API works is that the computation is first staged to a `Jaxpr`, and then an interpreter is run. The interpreter traverses the `Jaxpr`, and replaces JAX primitives (like `jax.lax.add_p`) with ones from [MAX's operation set](https://docs.modular.com/max/api/mojo/graph/ops/).

## State of coverage of JAX primitives

Keep in mind, even if a primitive is supported by a test, there may be missing usage patterns which cause errors which we haven't covered yet.

- [X] `lax.add_p`
- [X] `lax.mul_p`