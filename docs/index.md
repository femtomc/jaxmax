# `juju`, from JAX to MAX.

<p align="center">
<img width="300px" src="./assets/good_juju.png"/>
</p>

`juju` is a bit of compiler middleware bridging JAX to the world of MAX graphs. It allows:

- users to write JAX programs, lower those programs to MAX graphs, and execute those graphs on MAX-supported hardware, including CPUs, GPUs, and (later on), xPUs (whatever MAX supports).
- users to extend the Python JAX language (the primitives that JAX exposes to write numerical programs) with custom MAX kernels.

!!! danger
    This package is a proof-of-concept, and really early in development. Simple programs only for now! Tons of JAX primitives are missing lowering rules. It's not yet clear _how much of JAX_ will be fully supported (and how many extensions via MAX kernels will be added).

**Example:**

Using `juju` to transform and execute code with MAX.

```python 
import jax.numpy as jnp
from juju import jit

@jit
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)

print(jax_code(5, 10).to_numpy()) # -0.93009484
```

## Getting started

If you're just starting out, you probably want to use `juju.jit`. This is an API which allows you to execute JAX compatible Python programs using MAX.

::: juju.jit

::: juju.make_max_graph

## Going deeper: custom operations and primitives