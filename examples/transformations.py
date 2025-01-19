from juju import jit
from jax import grad, vmap, make_jaxpr
import jax.numpy as jnp


def f(x):
    return x**2


fn = grad(lambda x: jnp.mean(vmap(f)(x)))
print(make_jaxpr(fn)(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])))

loss = jit(grad(lambda x: jnp.mean(vmap(f)(x))))

print(loss(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])))
