# `juju`, from JAX to MAX.

<p align="center">
<img width="300px" src="./assets/good_juju.png"/>
</p>

`juju` is a bit of compiler middleware bridging (parts of) JAX to the world of MAX graphs. It allows:

- users to write JAX programs (see caveat below), lower those programs to MAX graphs, and execute those graphs on MAX-supported hardware, including CPUs, GPUs, and (later on), xPUs (whatever MAX supports).
- users to extend the Python JAX language (the primitives that JAX exposes to write numerical programs) with custom MAX kernels.

!!! danger "Danger, Will Robinson!"
    This package is a proof-of-concept, and very early in development. Simple programs only for now! It's not yet clear _how much of JAX_ will be fully supported (and how many extensions via MAX kernels will be added).
    
    JAX is a massive project, with tons of functionality! It's unlikely that this package will ever support _all of JAX_ (all JAX primitives, and device semantics). The goal is to support enough JAX to be dangerous, and to provide ways to easily extend the functionality of this package to support e.g. more of JAX, or to plug your own custom operations to define your own JAX-like language with compilation to MAX.

**Example:**

Using `juju` to transform and execute code with MAX.

```python exec="on" source="material-block"
import jax.numpy as jnp
from juju import jit

@jit
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)

print(jax_code(5, 10).to_numpy()) 
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

There are two steps to exposing custom operations to `juju`:

- Writing a MAX kernel using Mojo.
- Exposing the kernel to MAX, and providing the necessary information to JAX in the form of a new `Primitive`.

### Writing a MAX kernel

A MAX kernel takes the form of a Mojo source code file. [The MAX development team has kindly shared several of these kernels for study.](https://github.com/modular/max/tree/nightly/examples/custom_ops) Additionally, [this article](https://docs.modular.com/nightly/max/tutorials/build-custom-ops/) is worth reading to gain a general understanding of custom operations.

Let's examine a kernel, and imagine that we've placed this into a folder called `kernels/add_one.mojo`:

```mojo title="kernels/add_one.mojo"
import compiler
from utils.index import IndexList
from max.tensor import ManagedTensorSlice, foreach
from runtime.asyncrt import MojoCallContextPtr


@compiler.register("add_one", num_dps_outputs=1)
struct AddOneCustom:
    @staticmethod
    fn execute[
        # Parameter that if true, runs kernel synchronously in runtime
        synchronous: Bool,
        # e.g. "CUDA" or "CPU"
        target: StringLiteral,
    ](
        # as num_dps_outputs=1, the first argument is the "output"
        out: ManagedTensorSlice,
        # starting here are the list of inputs
        x: ManagedTensorSlice[out.type, out.rank],
        # the context is needed for some GPU calls
        ctx: MojoCallContextPtr,
    ):
        @parameter
        @always_inline
        fn func[width: Int](idx: IndexList[x.rank]) -> SIMD[x.type, width]:
            return x.load[width](idx) + 1

        foreach[func, synchronous, target](out, ctx)

    # You only need to implement this if you do not manually annotate
    # output shapes in the graph.
    @staticmethod
    fn shape(
        x: ManagedTensorSlice,
    ) raises -> IndexList[x.rank]:
        raise "NotImplemented"

```

Kernels are Mojo structures that are decorated with `@compiler.register`, and they contain a method called `execute` which contains the execution semantics of the kernel.

To expose the kernel as a MAX operation, the kernel needs to be placed into a Mojo package -- meaning we need a `kernels/__init__.mojo`:

```mojo title="kernels/__init__.mojo"
from .add_one import *
```

We can then ask `mojo` to compile the Mojo package into a `kernels.mojopkg`, which we can then use via MAX's Python API to give MAX access to the kernels:

```
mojo package kernels -o kernels.mojopkg
```

!!! important "Keep your kernels package up to date!"
    When implementing custom operations, make sure that the kernels package you're using is up-to-date! Otherwise, during graph loading, MAX will complain about being unable to find your kernel.

In the Python API, we can give access to the kernels by providing a `custom_extensions` argument to `engine.InferenceSession`:

```python
from max import engine 

engine.InferenceSession(
    custom_extensions="./kernels.mojopkg",
)
```

This is exactly how `juju` does this under the hood, and examining the code should provide further details.

### Exposing the kernel to JAX

Now, MAX is only one side of the coin. The other side is that we'd like to incorporate these computations in JAX source code. 

JAX allows users to extend JAX's program representations ([the Jaxpr](https://jax.readthedocs.io/en/latest/jaxpr.html)) by introducing new _primitives_, units of computation that accept and return arrays.

#### Interim on the `juju` pipeline

`juju` plugs into JAX in the following way:

- (**Tracing**) First, we use JAX to trace Python computations to produce Jaxprs. 
- (**Lowering**) Then, `juju` processes these Jaxprs with an interpreter to create MAX graphs.

Let's say we want to introduce a new primitive to JAX. The first **tracing** stage requires that the primitive communicate with JAX about the shapes and dtypes of the arrays it accepts as input, as well as the shapes and dtypes of the arrays it produces as output. As long as we tell JAX this information, it doesn't care about "what the primitive does". We'll call this information a `jax_abstract_evaluation_rule`.

The second **lowering** stage requires that we tell the `juju` interpreter how the primitive is going to be represented in the MAX graph. We'll call this information a `max_lowering_rule`.

To aid in the effort of coordination between JAX and MAX, `juju` exposes a function called `juju.Primitive`:

::: juju.Primitive

For instance, to use our `add_one` kernel, one would use the following patterns:

```python title="using_our_prim.py" exec="on" source="material-block"
from juju import Primitive, jit
from jax.core import ShapedArray
import jax.numpy as jnp
from max.graph import ops, TensorType

# Lowering rule to MAX, gets called by 
# juju's lowering interpreter.
def add_one_lowering(x, **params):
    return ops.custom(
        name="add_one", # needs to match your @compiler.register name
        values=[x],
        out_types=[TensorType(dtype=x.dtype, shape=x.tensor.shape)],
    )[0]

# Abstract evaluation rule for JAX, gets called
# by JAX when tracing a program to a Jaxpr.
def add_one_abstract(x, **params):
    return ShapedArray(x.shape, x.dtype)

# Register and coordinate everything, get a callable back.
add_one = Primitive(
    "add_one", # can be anything
    add_one_lowering, 
    add_one_abstract,
)

@jit
def jaxable_program(x):
    x = x * 2
    return add_one(x) # use the callable

# Execute your program using MAX.
print(jaxable_program(jnp.ones(10)).to_numpy())
```

The point being that `juju.Primitive` acts as a very convenient glue between JAX and MAX.