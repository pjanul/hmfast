import jax
import jax.numpy as jnp
from jax import lax

def newton_root(f, x0, tol=1e-8, max_iter=50):
    df = jax.grad(f)

    def cond_fn(state):
        x, i = state
        return (jnp.abs(f(x)) > tol) & (i < max_iter)

    def body_fn(state):
        x, i = state
        fx = f(x)
        dfx = df(x)
        x_new = x - fx / dfx
        return (x_new, i + 1)

    x_final, _ = lax.while_loop(cond_fn, body_fn, (x0, 0))
    return x_final
