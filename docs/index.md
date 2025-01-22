# `juju`: JAX to MAX!

<p align="center">
<img width="300px" src="./assets/good_juju.png"/>
</p>

`juju` is a bit of compiler middleware bridging JAX to the world of MAX graphs. It allows users to write JAX programs, lower those programs to MAX graphs, and execute those graphs on MAX-supported hardware.

**Example:**

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

It also exposes some tools which are designed to make it convenient to integrate custom MAX kernels into JAX-compatible Python functions.

## Getting started

If you're just starting out, you probably want to use `juju.jit`. This is an API which allows you to execute JAX compatible Python programs using MAX.

::: juju.jit

::: juju.make_max_graph

## Going deeper: custom operations and primitives