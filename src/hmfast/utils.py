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
    _C1_IA_ = 5e-14                # h^-2 Msun^-1 Mpc^3 (Hirata & Seljak 2004 / Joachimi et al. 2011 Eq. 6 normalization)

    # parameters entering in Stefan-Boltzmann constant sigma_B 
    _k_B_ = 1.3806504e-23
    _h_P_ = 6.62606896e-34
    _M_sun_ =  1.98855e30 # solar mass in kg

    _sigma_B_ = 5.6704004737209545e-08




# Newton's method root finder
def newton_solver(f, x0, tol=1e-8, max_iter=25):
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

def newton_root(f, x0, tol=1e-8, max_iter=25):
    def solve(f, x0):
        return newton_solver(f, x0, tol=tol, max_iter=max_iter)
    # For scalar roots, use the recommended tangent_solve
    tangent_solve = lambda g, y: y / g(1.0)
    return lax.custom_root(f, x0, solve, tangent_solve=tangent_solve)


# Dormand-Prince (DOPRI5) Butcher tableau -- order 5, embedded order-4 error estimate.
_DOPRI5_C = (1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0)
_DOPRI5_A = (
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
)
_DOPRI5_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84)
_DOPRI5_B4 = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40)


def dopri5_integrate(rhs, y0, x_start, x_max, rtol=1e-6, atol=1e-9, max_steps=200, max_h=None):
    """
    Adaptive-step embedded Dormand-Prince (order 5(4)) integration of
    dy/dx = rhs(x, y), y(x_start) = y0, from x_start to x_max.

    Every proposed step is accepted unconditionally; its embedded 4th-order
    error estimate is used only to size the *next* step to (rtol, atol).
    Runs inside a fixed-length ``lax.scan`` of ``max_steps`` proposals so it
    stays reverse-mode differentiable -- once x reaches x_max, remaining
    proposals are frozen (zero-length), so max_steps is a safety cap on
    resolution, not an accuracy control; rtol/atol control accuracy.

    Parameters
    ----------
    rhs : callable
        Right-hand side ``rhs(x, y) -> dy/dx``. ``y`` and its return
        value may be arbitrary JAX pytrees of matching structure.
    y0 : pytree
        Initial state at ``x_start``.
    x_start, x_max : float
        Integration bounds (``x_max`` may be less than ``x_start`` to
        integrate backward).
    rtol, atol : float
        Relative/absolute error tolerance used for step-size control.
    max_steps : int
        Static upper bound on the number of proposed steps; must be large
        enough that x reaches x_max well before it is exhausted.
    max_h : float or None
        Optional ceiling on the step size. The tolerance-driven step size
        can otherwise grow large enough on a smooth problem that the
        returned trajectory is too sparse to safely interpolate between
        nodes for callers reading off *interior* points, not just
        ``x_traj[-1]``; set this to bound node spacing for such callers.

    Returns
    -------
    x_traj : jnp.ndarray
        Up to ``max_steps + 1`` integration nodes, including ``x_start``;
        nodes after convergence repeat ``x_max``.
    y_traj : pytree
        State trajectory at each node in ``x_traj``, same pytree structure
        as ``y0`` with a leading axis of length ``max_steps + 1`` on every leaf.
    """
    direction = jnp.sign(x_max - x_start)
    h0 = (x_max - x_start) / max_steps
    if max_h is not None:
        h0 = direction * jnp.minimum(jnp.abs(h0), max_h)

    def combine(y, h, coeffs, ks):
        return jax.tree_util.tree_map(
            lambda y_leaf, *k_leaves: y_leaf + h * sum(c * k for c, k in zip(coeffs, k_leaves)),
            y, *ks,
        )

    def rms_norm(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        mean_sq = sum(jnp.sum(l ** 2) for l in leaves) / sum(l.size for l in leaves)
        return jnp.sqrt(mean_sq + 1e-300)  # avoid sqrt's singular grad at exactly 0

    def step(carry, _):
        x, y, h = carry
        remaining = x_max - x
        done = direction * remaining <= 0
        h_eff = jnp.where(done, 0.0, jnp.where(jnp.abs(h) > jnp.abs(remaining), remaining, h))

        ks = [rhs(x, y)]
        for a_row, c in zip(_DOPRI5_A, _DOPRI5_C):
            ks.append(rhs(x + c * h_eff, combine(y, h_eff, a_row, ks)))
        y5 = combine(y, h_eff, _DOPRI5_B5, ks)
        k7 = rhs(x + h_eff, y5)
        y4 = combine(y, h_eff, _DOPRI5_B4, ks + [k7])

        err_norm = rms_norm(jax.tree_util.tree_map(
            lambda a, b, y5_leaf: (a - b) / (atol + rtol * jnp.abs(y5_leaf)), y5, y4, y5
        ))
        factor = jnp.clip(0.9 * err_norm ** (-0.2), 0.2, 5.0)
        h_prop = h * factor
        if max_h is not None:
            h_prop = direction * jnp.minimum(jnp.abs(h_prop), max_h)
        h_new = jnp.where(done, h, h_prop)
        x_new = x + h_eff
        return (x_new, y5, h_new), (x_new, y5)

    _, (x_traj, y_traj) = lax.scan(step, (x_start, y0, h0), None, length=max_steps)
    x_traj = jnp.concatenate([jnp.asarray([x_start]), x_traj])
    y_traj = jax.tree_util.tree_map(
        lambda leaf0, traj: jnp.concatenate([jnp.asarray(leaf0)[None], traj]), y0, y_traj
    )
    return x_traj, y_traj



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


@jax.jit
def log_interp1d_extrap(x, xp, fp):
    """
    Interpolate positive 1D data in log-log space and extrapolate using the
    endpoint log-slopes.

    Parameters
    ----------
    x : float or jnp.ndarray
        Evaluation points. Must be strictly positive.
    xp : jnp.ndarray
        Monotonic sample points. Must be strictly positive.
    fp : jnp.ndarray
        Positive function values sampled on ``xp``.

    Returns
    -------
    jnp.ndarray
        Interpolated or extrapolated values with the same shape as ``x``.
    """
    x = jnp.asarray(x)
    xp = jnp.asarray(xp)
    fp = jnp.asarray(fp)

    log_x = jnp.log(x)
    log_xp = jnp.log(xp)
    log_fp = jnp.log(fp)

    left_slope = (log_fp[1] - log_fp[0]) / (log_xp[1] - log_xp[0])
    right_slope = (log_fp[-1] - log_fp[-2]) / (log_xp[-1] - log_xp[-2])

    log_f_interp = jnp.interp(log_x, log_xp, log_fp)
    log_f_left = log_fp[0] + left_slope * (log_x - log_xp[0])
    log_f_right = log_fp[-1] + right_slope * (log_x - log_xp[-1])

    log_f = jnp.where(
        x < xp[0],
        log_f_left,
        jnp.where(x > xp[-1], log_f_right, log_f_interp),
    )
    return jnp.exp(log_f)