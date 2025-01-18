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

## MAX graphs

What is a MAX graph? Let's inspect one:

```python
import jax.numpy as jnp
from jaxmax import max_graph

@max_graph
def jax_code(x, y):
    v = x + y
    v = v * v
    return jnp.sin(v)

print(jax_code(5, 10)) # -0.93009484
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

## State of coverage of JAX primitives

Keep in mind, even if a primitive is supported by a test, there may be missing usage patterns which cause errors which we haven't covered yet.

- [X] `lax.add_p`
- [X] `lax.mul_p`
- [X] `lax.sin_p`
- [X] `lax.convert_element_type_p`