# juju

<br>
<p align="center">
<img width="300px" src=".docs/assets/good_juju.png"/>
</p>

`juju` is a compiler from [JAX](https://github.com/jax-ml/jax) to [MAX](https://www.modular.com/max). It supports functionality which is designed to transform _JAX computations_ into [MAX computation graphs](https://docs.modular.com/max). These graphs can then be executed using MAX.

> [!CAUTION]
> This package is a proof-of-concept, and likely has sharp edges. Simple programs only for now! Tons of JAX primitives are missing lowering rules. It's not yet clear _how much of JAX_ will be fully supported (and how many extensions via MAX kernels will be added).

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

The compiler works as follows:
- First, the computation is first staged to a `Jaxpr`
- Then, an interpreter is run. The interpreter traverses the `Jaxpr`, and replaces JAX primitives (like `jax.lax.add_p`) with ones from [MAX's operation set](https://docs.modular.com/max/api/mojo/graph/ops/) to produce a MAX graph.
- The MAX graph can be loaded into a MAX execution engine (a `InferenceSession`) and executed.

In theory, one could define the functionality that we've all come to know and love from JAX, using MAX as a backend in place of XLA.

> [!WARNING]
> You can't invoke MAX computations _within a JAX computation which you `jax.jit`_ yet. In other words, you can't mix and max XLA and MAX in this package yet, and it's not clear if this will ever work.

## Getting started

First, [install `magic`](https://docs.modular.com/magic/). Then, clone this repository, and run `magic install` at the toplevel. This will setup your environment, which you can access via `magic shell`. You'll also want to run `magic run kernels` to build the custom MAX kernels provided as part of `juju`.

Inside the shell, you can run the example snippets using `python examples/basic.py` (for instance).


## MAX graphs

What is a MAX graph? Let's inspect one:

```python
import jax.numpy as jnp
from juju import make_max_graph

@make_max_graph
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)

print(jax_code(5, 10)) 
```

produces a textual object which looks like the following:

```
mo.graph @jax_code(%arg0: !mo.tensor<[], si32>, %arg1: !mo.tensor<[], si32>) -> !mo.tensor<[], f32> attributes {argument_names = ["input0", "input1"], result_names = ["output0"]} {
  %0 = mo.chain.create()
  %1 = rmo.add(%arg0, %arg1) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[], si32>
  %2 = rmo.mul(%1, %1) : (!mo.tensor<[], si32>, !mo.tensor<[], si32>) -> !mo.tensor<[], si32>
  %3 = mo.cast(%2) : (!mo.tensor<[], si32>) -> !mo.tensor<[], f32>
  %4 = rmo.mo.sin(%3) : (!mo.tensor<[], f32>) -> !mo.tensor<[], f32>
  mo.output %4 : !mo.tensor<[], f32>
}
```

This is a MAX graph, an intermediate representation which can be fed to [MAX's execution engine](https://docs.modular.com/max/api/mojo/engine/) to perform computations.

## Using `juju.jit`

We expose a very simple implementation of JIT functionality (similar to `jax.jit`) based on a JIT cache using [static Pytree structure](https://jax.readthedocs.io/en/latest/pytrees.html):

```python
from jax import grad
import jax.numpy as jnp
from juju import jit

@jit
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)

timing(jax_code)(5, 10) # 0.131628 s
timing(jax_code)(5, 10) # 0.000175 s 
```

The idea here is simple: a global MAX inference session is kept, and models are created and loaded into this session. We create a callable which executes these saved models, and we store this callable in a cache according to keys of the form `(your_hashable_callable, static_pytree_structure)`.

> [!WARNING]
> This is not nearly as featured as `jax.jit`. Indeed, some things you _cannot_ do with this now:
> * Invoke a `juju.jit` function inside of code which you want to lower to MAX.
> * No keyword arguments to specify static arguments.

## Composition with JAX transformations

Our approach is fully compositional with JAX transformations, meaning one can apply transformations like `jax.vmap` and `jax.grad` _before_ lowering the resulting computation to a MAX graph.

```python
from jax import grad
import jax.numpy as jnp
from juju import jit

@jit
@grad
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)


print(jax_code(5.0, 10.0).to_numpy()) # 11.019581
```

## Extending `juju` with custom MAX operations

[Full example here.](https://github.com/femtomc/juju/blob/main/examples/custom_primitives.py)

[MAX supports a user-extensible operation set](https://docs.modular.com/nightly/max/tutorials/build-custom-ops/), and `juju` allows you to expose these operations into JAX computations which you intend to lower to MAX.

To start, [one writes a kernel in Mojo](https://github.com/femtomc/juju/blob/main/src/juju/kernels/mandelbrot.mojo) and registers it with the MAX engine. 

In `juju`, this part looks like placing a `your_kernel.mojo` file into the `src/juju/kernels` Mojo sub-package, and exporting your kernel in `src/juju/kernels/__init__.mojo`. Once you've done this, you can run `magic run kernels` to create a new kernel package (`./kernels.mojopkg`) for use via MAX's Python API, and through `juju`. 

Now, `juju` exposes a registration function called `Primitive`. 

```python
# Lowering rule to MAX operation.
def mandelbrot_max_lowering_rule(**params):
    min_x = params["min_x"]
    min_y = params["min_y"]
    scale_x = params["scale_x"]
    scale_y = params["scale_y"]
    width = params["width"]
    height = params["height"]
    max_iterations = params["max_iterations"]
    output_dtype = DType.int32
    return ops.custom(
        name="mandelbrot",
        values=[
            ops.constant(min_x, dtype=DType.float32),
            ops.constant(min_y, dtype=DType.float32),
            ops.constant(scale_x, dtype=DType.float32),
            ops.constant(scale_y, dtype=DType.float32),
            ops.constant(max_iterations, dtype=DType.int32),
        ],
        out_types=[TensorType(dtype=output_dtype, shape=[height, width])],
    )[0].tensor

# Tell JAX how to interpret the semantics of the primitive in terms
# of types that it understands.
def mandelbrot_abstract_eval(**params):
    height = params["height"]
    width = params["width"]
    return ShapedArray((height, width), jnp.int32)

# Use Primitive to register the lowering and abstract evaluation rules.
mandelbrot = Primitive(
    "mandelbrot", # name of the kernel that MAX knows about.
    mandelbrot_max_lowering_rule,
    mandelbrot_abstract_eval,
    multiple_results=False,
)
```

Having done this, you can use the new primitive in Python JAX computations:

```python
def compute_mandelbrot():
    WIDTH = 30
    HEIGHT = 30
    MAX_ITERATIONS = 500
    MIN_X = -1.5
    MAX_X = 0.7
    MIN_Y = -1.12
    MAX_Y = 1.12
    scale_x = (MAX_X - MIN_X) / WIDTH
    scale_y = (MAX_Y - MIN_Y) / HEIGHT
    return mandelbrot(
        min_x=MIN_X,
        min_y=MIN_Y,
        scale_x=scale_x,
        scale_y=scale_y,
        width=WIDTH,
        height=HEIGHT,
        max_iterations=MAX_ITERATIONS,
    )
```

Now, if you print the MAX graph for this computation, you'll see something like this:

```
mo.graph @compute_mandelbrot() -> !mo.tensor<[15, 15], si32> attributes {argument_names = [], result_names = ["output0"]} {
  %0 = mo.chain.create()
  %1 = mo.constant {value = #M.dense_array<-1.500000e+00> : tensor<f32>} : !mo.tensor<[], f32>
  %2 = mo.constant {value = #M.dense_array<-1.120000e+00> : tensor<f32>} : !mo.tensor<[], f32>
  %3 = mo.constant {value = #M.dense_array<0.146666661> : tensor<f32>} : !mo.tensor<[], f32>
  %4 = mo.constant {value = #M.dense_array<0.149333328> : tensor<f32>} : !mo.tensor<[], f32>
  %5 = mo.constant {value = #M.dense_array<100> : tensor<si32>} : !mo.tensor<[], si32>
  %6 = mo.custom {symbol = "mandelbrot"}(%1, %2, %3, %4, %5) : (!mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], si32>) -> !mo.tensor<[15, 15], si32>
  mo.output %6 : !mo.tensor<[15, 15], si32>
}
```
we can see our custom operation on line `%6`, along with its inputs and outputs.

Executing the computation via MAX:
```python
print(jit(compute_mandelbrot)().to_numpy())
```

produces:
```
[[  2   2   3   3   3   3   3   3   4   6   4   3   3   2   2]
 [  3   3   3   3   3   3   4   4   5   8  10   4   4   3   2]
 [  3   3   3   3   3   4   4   5   7 100  23   5   5   4   3]
 [  3   3   3   4   4   5  18   9  12 100  18  12   7   6   3]
 [  3   3   4   5   5   6  10 100 100 100 100 100 100   7   4]
 [  4   6   8   8   7   8 100 100 100 100 100 100 100  17   4]
 [  5   7  14 100 100  13 100 100 100 100 100 100 100  51   4]
 [  7  24 100 100 100 100 100 100 100 100 100 100 100   7   4]
 [  7  24 100 100 100 100 100 100 100 100 100 100 100   7   4]
 [  5   7  14 100 100  13 100 100 100 100 100 100 100  51   4]
 [  4   6   8   8   7   8 100 100 100 100 100 100 100  17   4]
 [  3   3   4   5   5   6  10 100 100 100 100 100 100   7   4]
 [  3   3   3   4   4   5  18   9  12 100  18  12   7   6   3]
 [  3   3   3   3   3   4   4   5   7 100  23   5   5   4   3]
 [  3   3   3   3   3   3   4   4   5   8  10   4   4   3   2]]
```


## State of coverage of JAX primitives

Keep in mind, even if a primitive is supported by a test, there may be missing usage patterns which cause errors which we haven't covered yet.

- [X] `lax.add_p`
- [X] `lax.mul_p`
- [X] `lax.sin_p`
- [X] `lax.cos_p`
- [X] `lax.neg_p`
- [X] `lax.abs_p`
- [X] `lax.convert_element_type_p`
- [X] `ad_util.add_any_p`