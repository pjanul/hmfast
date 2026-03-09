import jax
import jax.numpy as jnp
from jax import lax
from functools import partial


# This file is primarily to implement functions/units that are needed for this package but are not implemented in JAX

# Create a central location for all constants
class Const:
    #_c_km_s_ = 299792.458  # speed of light in km/s
    #_h_J_s_ = 6.626070040e-34  # Planck's constant
    #_kB_J_K_ = 1.38064852e-23  # Boltzmann constant

    _c_ = 2.99792458e8      # c in m/s 
    _Mpc_over_m_ = 3.085677581282e22  # conversion factor from meters to megaparsecs 
    _Gyr_over_Mpc_ = 3.06601394e2 # conversion factor from megaparsecs to gigayears
    _G_ = 6.67428e-11             # Newton constant in m^3/Kg/s^2 
    _eV_ = 1.602176487e-19        # 1 eV expressed in J 
    _sigma_T_ = 6.6524587e-29     # Thomson cross-section in m^2
    _m_e_ = 9.1093837015e-31      # Electron mass in kg
    _m_p_ = 1.67262192369e-27     # Proton mass in kg
    _L_sun_ = 3.828e26            # Solar luminosity in Watts (1 kg·m²/s³)

    # parameters entering in Stefan-Boltzmann constant sigma_B 
    _k_B_ = 1.3806504e-23
    _h_P_ = 6.62606896e-34
    _M_sun_ =  1.98855e30 # solar mass in kg

    _sigma_B_ = 5.6704004737209545e-08




# Newton's method root finder
def newton_solver(f, x0, tol=1e-8, max_iter=50):
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

def newton_root(f, x0, tol=1e-8, max_iter=50):
    def solve(f, x0):
        return newton_solver(f, x0, tol=tol, max_iter=max_iter)
    # For scalar roots, use the recommended tangent_solve
    tangent_solve = lambda g, y: y / g(1.0)
    return lax.custom_root(f, x0, solve, tangent_solve=tangent_solve)



# Lambert W function. As of the writing of this comment, it is not yet implemented in JAX
def _real_lambertw_recursion(w: jax.Array, x: jax.Array) -> jax.Array:
    return w / (1+w) * (1+jnp.log(x / w))


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def _lambertwk0(x, max_steps=5):
    # See https://en.wikipedia.org/wiki/Lambert_W_function#Numerical_evaluation
    w_0 = jax.lax.select(
        x > jnp.e,
        jnp.log(x) - jnp.log(jnp.log(x)),
        x / jnp.e
    )
    w_0 = jax.lax.select(
        x > 0,
        w_0,
        jnp.e * x / (1 + jnp.e * x + jnp.sqrt(1 + jnp.e * x)) * jnp.log(
            1 + jnp.sqrt(1 + jnp.e * x))
    )
    
    w, _ = jax.lax.scan(
        lambda carry, _: (_real_lambertw_recursion(carry, x),)*2,
        w_0,
        xs=None, length=max_steps
    )
    
    w = jax.lax.select(
        jnp.isclose(x, 0.0),
        0.0,
        w
    )
        
    return w


@_lambertwk0.defjvp
def _lambertw_jvp(max_steps, primals, tangents):
    # Note: All branches for lambert W satisfy this JVP.
    x, = primals
    t, = tangents

    y = _lambertwk0(x, max_steps)
    dydx = 1 / (x + jnp.exp(y))

    jvp = jax.lax.select(
        jnp.isclose(x, -1/jnp.e),
        jnp.nan,
        dydx * t
    )

    return y, jvp


@jnp.vectorize
def lambertw(x, k=0, max_steps=5):
    if k != 0:
        raise NotImplementedError()

    return _lambertwk0(x, max_steps=max_steps)