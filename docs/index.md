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

To get started with `juju`, you'll need to follow these steps:

- First, [install `magic`](https://docs.modular.com/magic/), the package and compiler manager for MAX and Mojo.
- Then, clone this repository, and run `magic install` at the toplevel. This will setup your environment, which you can access via `magic shell`. 
- Then, run `magic run kernels` to build the custom MAX kernels provided as part of `juju`.

## Basic APIs

To start out, let's examine basic APIs which allow you to execute functions using MAX, and create MAX graphs. 

::: juju.jit

::: juju.make_max_graph

## Custom operations and primitives

A very nice feature of MAX is that [the operation set is extensible](https://docs.modular.com/nightly/max/tutorials/build-custom-ops/), and [the language for authoring operations is Mojo](https://www.modular.com/mojo), a language with high-level ergonomics (compared to CUDA, for instance).

As a result, extending the operation set with new GPU computations is much more approachable than extending XLA with custom CUDA computations, and can be performed without leaving the `juju` project or introducing external compilers (besides the Mojo compiler, which is accessed via `magic`).

::: juju.Primitive