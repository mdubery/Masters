"""Compiled version of the single-planet evolution.

This repeats mass_fraction_evolver.RK45_driver for the case with no stored
time series and no stellar-evolution tracks. The numerical method and the
mass-loss formulae are the same. It is used when only the final radius and
envelope fraction are needed.
"""

import numpy as np
from numba import njit

from constants import (
    pi, stefan, kappa_0, G, gamma, Delta_ab, mu, m_H, k_B,
    alpha, beta, M_sun, L_sun, AU, T_eq_earth, M_earth, R_earth,
    eta_0, Myr, P_TRANSIT, P_XUV,
)

ATMOSPHERE_FLOOR = 1e-4
STRIPPED_ATMOSPHERE_THRESHOLD = 5e-4

# Exponents that appear repeatedly in the structure equations.
_INV_GAMMA_M1 = 1.0 / (gamma - 1.0)
_INV_ALPHA_P1 = 1.0 / (alpha + 1.0)

# Out-of-table Aiolos continuation. Must stay consistent with aiolos_mass_loss.py.
_R_EARTH_CM = R_earth * 1.0e2
_LN10 = np.log(10.0)


# Envelope structure. These functions follow planet_structure.py.

@njit(cache=True)
def _I2(dR_Rc):
    if dR_Rc <= 0.0:
        return np.nan
    log_dR_Rc = np.log10(dR_Rc)
    log_I2 = (
        np.log(1.2598238012325735 + np.exp(-2.4535812359864062 * 1.0798633578446974 * log_dR_Rc))
        / -1.0798633578446974
        - 4.8814961274562990e-01
    )
    return 10.0 ** log_I2


@njit(cache=True)
def _I2_I1(dR_Rc):
    if dR_Rc <= 0.0:
        return np.nan
    log_dR_Rc = np.log10(dR_Rc)
    log_I2_I1 = (
        -7.6521840215423576e-01
        / (1.0 + np.exp(-(log_dR_Rc - 5.7429970641208375e-02) / 6.4338705851296174e-01))
        ** 1.8214254374336605
    )
    return 10.0 ** log_I2_I1


@njit(cache=True)
def _R_rcb_equation(
    R_rcb, T_eq, c_s_squared, KH_timescale_seconds,
    M_core_kg, R_core_meters, X, use_Lcore,
):
    if (not np.isfinite(X)) or X <= 0.0 or R_rcb <= 0.0:
        return np.nan

    psi = 1.0 if use_Lcore else 0.0

    c1 = 4.0 * pi * mu * m_H / (M_core_kg * k_B)
    c2 = (Delta_ab * G * M_core_kg / c_s_squared) ** _INV_GAMMA_M1
    c3 = (
        1088.0 * pi * stefan * (T_eq ** (3.0 - alpha - beta))
        * KH_timescale_seconds
        / (3.0 * kappa_0 * M_core_kg * (17.0 * X + psi))
    ) ** _INV_ALPHA_P1

    c = c1 * c2 * c3
    dR = (R_rcb / R_core_meters) - 1.0
    return (
        c
        * (R_rcb ** 3)
        * _I2(dR)
        * (R_rcb ** (-_INV_GAMMA_M1))
        * ((_I2_I1(dR) * R_rcb) ** _INV_ALPHA_P1)
        - X
    )


@njit(cache=True)
def _brentq_R_rcb(
    lo, hi, T_eq, c_s_squared, KH_timescale_seconds,
    M_core_kg, R_core_meters, X, use_Lcore,
    xtol=1e-12, maxiter=200,
):
    """Find the radiative-convective boundary by bisection.

    Returns 0 if the bracket has no root.
    """
    a = lo
    b = hi
    fa = _R_rcb_equation(
        a, T_eq, c_s_squared, KH_timescale_seconds,
        M_core_kg, R_core_meters, X, use_Lcore,
    )
    fb = _R_rcb_equation(
        b, T_eq, c_s_squared, KH_timescale_seconds,
        M_core_kg, R_core_meters, X, use_Lcore,
    )
    if (not np.isfinite(fa)) or (not np.isfinite(fb)) or fa * fb > 0.0:
        return 0.0

    for _ in range(maxiter):
        mid = 0.5 * (a + b)
        if np.abs(b - a) < xtol * (1.0 + np.abs(mid)):
            return mid
        fm = _R_rcb_equation(
            mid, T_eq, c_s_squared, KH_timescale_seconds,
            M_core_kg, R_core_meters, X, use_Lcore,
        )
        if not np.isfinite(fm) or fm == 0.0:
            return mid if np.isfinite(fm) else 0.0
        if fa * fm < 0.0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
    return 0.5 * (a + b)


@njit(cache=True)
def _isothermal_attached_radius(R_rcb, P_level, P_rcb, Rbondi):
    """Isothermal photosphere, kept between the boundary and the Bondi radius."""
    if R_rcb <= 0.0 or Rbondi <= 0.0 or P_rcb <= 0.0 or P_level <= 0.0:
        return R_rcb
    denom = 1.0 + 0.5 * R_rcb * np.log(P_level / P_rcb) / Rbondi
    if (not np.isfinite(denom)) or denom <= 0.0:
        return max(R_rcb, Rbondi)
    R = R_rcb / denom
    if (not np.isfinite(R)) or R <= 0.0:
        return R_rcb
    if R < R_rcb:
        return R_rcb
    if R > Rbondi:
        return Rbondi
    return R


@njit(cache=True)
def _solve_structure(
    T_eq, c_s_squared, KH_timescale_seconds,
    M_core_kg, R_core_meters, X, R_guess_earth, use_Lcore,
):
    """Envelope structure for one age and envelope fraction.

    Returns (R_rcb, Rho_rcb, R_transit, R_xuv) in metres and kg m^-3.
    Returns zeros if no boundary radius is found.
    """
    lo = R_core_meters * 1.0001
    max_hi = 500.0 * R_earth
    if R_guess_earth <= 0.0:
        hi = max_hi
    else:
        hi = min(max_hi, R_earth * (1.0 + R_guess_earth))

    f_lo = _R_rcb_equation(
        lo, T_eq, c_s_squared, KH_timescale_seconds,
        M_core_kg, R_core_meters, X, use_Lcore,
    )
    f_hi = _R_rcb_equation(
        hi, T_eq, c_s_squared, KH_timescale_seconds,
        M_core_kg, R_core_meters, X, use_Lcore,
    )
    while (
        np.isfinite(f_lo)
        and np.isfinite(f_hi)
        and (f_lo * f_hi > 0.0)
        and hi < max_hi
    ):
        hi = min(max_hi, 2.0 * hi)
        f_hi = _R_rcb_equation(
            hi, T_eq, c_s_squared, KH_timescale_seconds,
            M_core_kg, R_core_meters, X, use_Lcore,
        )

    if (not np.isfinite(f_lo)) or (not np.isfinite(f_hi)) or (f_lo * f_hi > 0.0):
        return 0.0, 0.0, 0.0, 0.0

    R_rcb = _brentq_R_rcb(
        lo, hi, T_eq, c_s_squared, KH_timescale_seconds,
        M_core_kg, R_core_meters, X, use_Lcore,
    )
    if R_rcb == 0.0:
        return 0.0, 0.0, 0.0, 0.0

    psi = 1.0 if use_Lcore else 0.0
    Rho_rcb = (mu * m_H / k_B) * (
        (
            _I2_I1((R_rcb / R_core_meters) - 1.0)
            * 1088.0 * pi * stefan
            * (T_eq ** (3.0 - alpha - beta))
            * KH_timescale_seconds
            * R_rcb
            / (3.0 * kappa_0 * M_core_kg * (17.0 * X + psi))
        ) ** _INV_ALPHA_P1
    )

    P_rcb = c_s_squared * Rho_rcb
    Rbondi = G * M_core_kg * (1.0 + X) / (2.0 * c_s_squared)
    # Pressures are in pascals: 20 mbar for the transit radius, 1 nbar for the XUV radius.
    R_transit = _isothermal_attached_radius(R_rcb, P_TRANSIT, P_rcb, Rbondi)
    R_xuv = _isothermal_attached_radius(R_rcb, P_XUV, P_rcb, Rbondi)
    return R_rcb, Rho_rcb, R_transit, R_xuv


@njit(cache=True)
def _calculate_T_eq(M_star, a):
    """Equilibrium temperature from L proportional to M_star^4.5. a is in AU."""
    return T_eq_earth * (1.0 / a) ** 0.5 * (M_star) ** (0.25 * 4.5)


@njit(cache=True)
def calculate_R_core_numba(composition, M_core):
    """Core radius in Earth radii. Same Fortney et al. (2007) fit as calculate_R_core."""
    logM = np.log10(M_core)
    logM2 = logM * logM
    if composition <= 0.0:
        imf = -composition
        return (
            (0.0912 * imf + 0.1603) * logM2
            + (0.3330 * imf + 0.7387) * logM
            + (0.4639 * imf + 1.1193)
        )
    rmf = 1.0 - composition
    return (
        (0.0592 * rmf + 0.0975) * logM2
        + (0.2337 * rmf + 0.4938) * logM
        + (0.3102 * rmf + 0.7932)
    )


# Four-dimensional linear interpolation of the Aiolos grid.

@njit(cache=True)
def _locate_and_weights(axis, x):
    """Index and weight for linear interpolation along one axis.

    Outside the axis the weight falls outside [0, 1], which continues the
    end interval in a straight line. That matches the regular-grid
    interpolant used while the Aiolos tables are built.
    """
    n = axis.shape[0]
    if x <= axis[0]:
        return 0, (x - axis[0]) / (axis[1] - axis[0])
    if x >= axis[n - 1]:
        return n - 2, (x - axis[n - 2]) / (axis[n - 1] - axis[n - 2])
    # Binary search for rightmost axis[i] <= x
    lo = 0
    hi = n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if axis[mid] <= x:
            lo = mid
        else:
            hi = mid
    i = lo
    t = (x - axis[i]) / (axis[i + 1] - axis[i])
    return i, t


@njit(cache=True)
def _interp4d(ax0, ax1, ax2, ax3, values, x0, x1, x2, x3):
    i0, t0 = _locate_and_weights(ax0, x0)
    i1, t1 = _locate_and_weights(ax1, x1)
    i2, t2 = _locate_and_weights(ax2, x2)
    i3, t3 = _locate_and_weights(ax3, x3)

    # Trilinear/quadrilinear over the 16 corners
    c0000 = values[i0, i1, i2, i3]
    c0001 = values[i0, i1, i2, i3 + 1]
    c0010 = values[i0, i1, i2 + 1, i3]
    c0011 = values[i0, i1, i2 + 1, i3 + 1]
    c0100 = values[i0, i1 + 1, i2, i3]
    c0101 = values[i0, i1 + 1, i2, i3 + 1]
    c0110 = values[i0, i1 + 1, i2 + 1, i3]
    c0111 = values[i0, i1 + 1, i2 + 1, i3 + 1]
    c1000 = values[i0 + 1, i1, i2, i3]
    c1001 = values[i0 + 1, i1, i2, i3 + 1]
    c1010 = values[i0 + 1, i1, i2 + 1, i3]
    c1011 = values[i0 + 1, i1, i2 + 1, i3 + 1]
    c1100 = values[i0 + 1, i1 + 1, i2, i3]
    c1101 = values[i0 + 1, i1 + 1, i2, i3 + 1]
    c1110 = values[i0 + 1, i1 + 1, i2 + 1, i3]
    c1111 = values[i0 + 1, i1 + 1, i2 + 1, i3 + 1]

    c000 = c0000 * (1.0 - t3) + c0001 * t3
    c001 = c0010 * (1.0 - t3) + c0011 * t3
    c010 = c0100 * (1.0 - t3) + c0101 * t3
    c011 = c0110 * (1.0 - t3) + c0111 * t3
    c100 = c1000 * (1.0 - t3) + c1001 * t3
    c101 = c1010 * (1.0 - t3) + c1011 * t3
    c110 = c1100 * (1.0 - t3) + c1101 * t3
    c111 = c1110 * (1.0 - t3) + c1111 * t3

    c00 = c000 * (1.0 - t2) + c001 * t2
    c01 = c010 * (1.0 - t2) + c011 * t2
    c10 = c100 * (1.0 - t2) + c101 * t2
    c11 = c110 * (1.0 - t2) + c111 * t2

    c0 = c00 * (1.0 - t1) + c01 * t1
    c1 = c10 * (1.0 - t1) + c11 * t1
    return c0 * (1.0 - t0) + c1 * t0


@njit(cache=True)
def _clip(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


@njit(cache=True)
def _axis_overflow(x, ax):
    o = 0.0
    if x < ax[0]:
        o = ax[0] - x
    elif x > ax[-1]:
        o = x - ax[-1]
    return o


@njit(cache=True)
def _R_trans_earth(mass_earth, Teq_k):
    return 4.2 * (mass_earth / 5.0) ** 0.85 * (Teq_k / 1000.0) ** (-1.05)


@njit(cache=True)
def _R_sonic_earth(mass_earth, Teq_k):
    """Isothermal sonic radius GM / (2 c_s^2), in Earth radii."""
    c_s_squared = k_B * Teq_k / (mu * m_H)
    return G * (mass_earth * M_earth) / (2.0 * c_s_squared) / R_earth


@njit(cache=True)
def _aiolos_mdot(
    planet_mass_earth, teq_k, transit_radius_cm, luv_erg_s,
    log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
):
    """Mass-loss rate in g/s from the Aiolos grid.

    Inside the table this is linear interpolation. Outside it, the rate is
    energy-limited below R_trans and a Parker wind at and above R_trans.
    """
    logM = np.log10(planet_mass_earth)
    logT = np.log10(teq_k)
    logR = np.log10(transit_radius_cm)
    logL = np.log10(luv_erg_s)

    log_lin = _interp4d(
        log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
        logM, logT, logR, logL,
    )

    o = _axis_overflow(logM, log_mass_ax)
    o = max(o, _axis_overflow(logT, log_teq_ax))
    o = max(o, _axis_overflow(logR, log_R_ax))
    o = max(o, _axis_overflow(logL, log_Luv_ax))
    if o <= 0.0:
        return 10.0 ** log_lin

    logMc = _clip(logM, log_mass_ax[0], log_mass_ax[-1])
    logTc = _clip(logT, log_teq_ax[0], log_teq_ax[-1])
    logRc = _clip(logR, log_R_ax[0], log_R_ax[-1])
    logLc = _clip(logL, log_Luv_ax[0], log_Luv_ax[-1])
    log_ref = _interp4d(
        log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
        logMc, logTc, logRc, logLc,
    )

    dlog_el = (
        -(logM - logMc) + 4.0 * (logT - logTc)
        + 3.0 * (logR - logRc) + (logL - logLc)
    )

    M = 10.0 ** logM
    T = 10.0 ** logT
    R = 10.0 ** logR / _R_EARTH_CM
    if R < 1.0e-6:
        R = 1.0e-6
    M_ref = 10.0 ** logMc
    T_ref = 10.0 ** logTc
    R_ref = 10.0 ** logRc / _R_EARTH_CM
    if R_ref < 1.0e-6:
        R_ref = 1.0e-6
    Rs = _R_sonic_earth(M, T)
    Rs_ref = _R_sonic_earth(M_ref, T_ref)
    dlog_parker = (
        2.0 * np.log10(Rs / Rs_ref)
        + 0.5 * np.log10(T / T_ref)
        + (-2.0 * Rs / R + 2.0 * Rs_ref / R_ref) / _LN10
    )

    Rtr = _R_trans_earth(M, T)
    if Rtr < 1.0e-12:
        Rtr = 1.0e-12
    x = np.log10(R / Rtr)
    if x >= 0.0:
        w = 1.0
    else:
        w = 0.0

    log_phys = log_ref + (1.0 - w) * dlog_el + w * dlog_parker
    return 10.0 ** log_phys


# Mass-loss timescale and dX/dt = -X / t_X.

@njit(cache=True)
def _calculate_tX(
    t, X, M_core, M_star, a, R_core, initial_KH_timescale, R_guess,
    use_Lcore, use_aiolos,
    log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
):
    M_core_kg = M_core * M_earth
    a_meters = a * AU
    t_seconds = t * Myr
    R_core_meters = R_core * R_earth
    M_env_kg = X * M_core_kg
    t_sat = 100.0 * Myr * (M_star ** (-1.0))

    T_eq = _calculate_T_eq(M_star, a)
    c_s_squared = (k_B * T_eq) / (mu * m_H)

    if t < initial_KH_timescale:
        KH_seconds = initial_KH_timescale * Myr
    else:
        KH_seconds = t * Myr

    R_rcb, Rho_rcb, R_transit, R_xuv = _solve_structure(
        T_eq, c_s_squared, KH_seconds,
        M_core_kg, R_core_meters, X, R_guess, use_Lcore,
    )
    if R_rcb == 0.0:
        return 0.0

    # X-ray luminosity: saturated, then declining as t^-1.3.
    L_sat = 10.0 ** (-3.50) * L_sun * (M_star ** 0.5)
    if t_seconds < t_sat:
        Lx = L_sat
    else:
        Lx = L_sat * (t_seconds / t_sat) ** (-1.3)

    if use_aiolos:
        M_env_dot = _aiolos_mdot(
            M_core * (1.0 + X),
            T_eq,
            R_transit * 1e2,
            Lx * 1e7,
            log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
        ) / 1000.0  # g/s to kg/s
    else:
        escape_velocity = np.sqrt(2.0 * G * M_core_kg / R_core_meters) * 0.001
        eta = eta_0 * (escape_velocity / 23.0) ** (-0.42)
        # Energy-limited rate. R_transit is the optical photosphere.
        M_env_dot = eta * R_transit ** 3 * Lx / (
            4.0 * a_meters * a_meters * G * M_core_kg
        )

    return (M_env_kg / M_env_dot) / Myr


@njit(cache=True)
def _dXdt(
    t, X, M_core, M_star, a, R_core, initial_KH_timescale, R_guess,
    use_Lcore, use_aiolos,
    log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
):
    tX = _calculate_tX(
        t, X, M_core, M_star, a, R_core, initial_KH_timescale, R_guess,
        use_Lcore, use_aiolos,
        log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
    )
    if tX == 0.0:
        return 0.0
    return -X / tX


# Cash-Karp step and the adaptive timestep controller.

@njit(cache=True)
def _rkck_step(
    t, var_i, dt, k1,
    M_core, M_star, a, R_core, initial_KH_timescale, R_guess,
    use_Lcore, use_aiolos,
    log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
):
    """One Cash-Karp step.

    Returns (fifth-order X, error estimate, status). status is 0 for a
    successful step, 1 if the trial envelope fraction went negative, and
    2 if the atmosphere structure could not be solved.
    """
    var1 = var_i + 0.2 * k1
    if var1 < 0.0:
        return 0.0, 0.0, 1

    k2 = _dXdt(
        t + 0.2 * dt, var1, M_core, M_star, a, R_core,
        initial_KH_timescale, R_guess, use_Lcore, use_aiolos,
        log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
    )
    if k2 == 0.0:
        return 0.0, 0.0, 2
    k2 *= dt

    var2 = var_i + (3.0 / 40.0) * k1 + (9.0 / 40.0) * k2
    if var2 < 0.0:
        return 0.0, 0.0, 1

    k3 = _dXdt(
        t + 0.3 * dt, var2, M_core, M_star, a, R_core,
        initial_KH_timescale, R_guess, use_Lcore, use_aiolos,
        log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
    )
    if k3 == 0.0:
        return 0.0, 0.0, 2
    k3 *= dt

    var3 = var_i + (3.0 / 10.0) * k1 - (9.0 / 10.0) * k2 + (6.0 / 5.0) * k3
    if var3 < 0.0:
        return 0.0, 0.0, 1

    k4 = _dXdt(
        t + 0.6 * dt, var3, M_core, M_star, a, R_core,
        initial_KH_timescale, R_guess, use_Lcore, use_aiolos,
        log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
    )
    if k4 == 0.0:
        return 0.0, 0.0, 2
    k4 *= dt

    var4 = var_i - (11.0 / 54.0) * k1 + (5.0 / 2.0) * k2 - (70.0 / 27.0) * k3 + (35.0 / 27.0) * k4
    if var4 < 0.0:
        return 0.0, 0.0, 1

    k5 = _dXdt(
        t + dt, var4, M_core, M_star, a, R_core,
        initial_KH_timescale, R_guess, use_Lcore, use_aiolos,
        log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
    )
    if k5 == 0.0:
        return 0.0, 0.0, 2
    k5 *= dt

    var5 = (
        var_i
        + (1631.0 / 55296.0) * k1
        + (175.0 / 512.0) * k2
        + (575.0 / 13824.0) * k3
        + (44275.0 / 110592.0) * k4
        + (253.0 / 4096.0) * k5
    )
    if var5 < 0.0:
        return 0.0, 0.0, 1

    k6 = _dXdt(
        t + (7.0 / 8.0) * dt, var5, M_core, M_star, a, R_core,
        initial_KH_timescale, R_guess, use_Lcore, use_aiolos,
        log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
    )
    if k6 == 0.0:
        return 0.0, 0.0, 2
    k6 *= dt

    RK5 = var_i + (37.0 / 378.0) * k1 + (250.0 / 621.0) * k3 + (125.0 / 594.0) * k4 + (512.0 / 1771.0) * k6
    RK4 = (
        var_i
        + (2825.0 / 27648.0) * k1
        + (18575.0 / 48384.0) * k3
        + (13525.0 / 55296.0) * k4
        + (277.0 / 14336.0) * k5
        + 0.25 * k6
    )
    if RK5 < 0.0:
        return 0.0, 0.0, 1
    return RK5, RK5 - RK4, 0


@njit(cache=True)
def _step_control(
    t, var_i, dt_try, accuracy,
    M_core, M_star, a, R_core, initial_KH_timescale, R_guess,
    use_Lcore, use_aiolos,
    log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
):
    """Choose a timestep and take one accepted step.

    Returns (t_new, X_new, dt_next, status). status is 0 for an accepted
    step, 1 if the atmosphere structure could not be solved, and 2 if the
    timestep collapsed while the envelope was still losing mass. In that
    last case the envelope is treated as stripped.
    """
    dydx = _dXdt(
        t, var_i, M_core, M_star, a, R_core, initial_KH_timescale, R_guess,
        use_Lcore, use_aiolos,
        log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
    )
    if dydx == 0.0:
        return 0.0, 0.0, 0.0, 1

    yscal = abs(var_i) + abs(dydx * dt_try) + 1e-3
    dt = dt_try

    while True:
        k1 = dydx * dt
        var_new, var_err, st = _rkck_step(
            t, var_i, dt, k1,
            M_core, M_star, a, R_core, initial_KH_timescale, R_guess,
            use_Lcore, use_aiolos,
            log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
        )
        if st == 1:
            dt *= 0.01
            if t + dt == t:
                if dydx < 0.0:
                    return t, ATMOSPHERE_FLOOR, 0.0, 2
                return 0.0, 0.0, 0.0, 1
            continue
        if st == 2:
            return 0.0, 0.0, 0.0, 1

        err_max = abs(var_err / yscal) / accuracy
        if err_max <= 1.0:
            break

        dt_new = 0.9 * dt / (err_max ** 0.25)
        dt = max(dt_new, 0.1 * dt)
        if t + dt == t:
            if dydx < 0.0:
                return t, ATMOSPHERE_FLOOR, 0.0, 2
            return 0.0, 0.0, 0.0, 1

    if err_max < 2e-4:
        dt_next = 5.0 * dt
    else:
        dt_next = 0.9 * (dt / err_max) ** 0.2

    return t + dt, var_new, dt_next, 0


@njit(cache=True)
def evolve_planet_numba(
    t_start, t_stop, dt_try, accuracy, initial_X,
    composition, M_core, period, M_star, initial_KH_timescale,
    use_Lcore, use_aiolos,
    log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
):
    """Final transit radius in Earth radii and final envelope fraction.

    Returns (0, 0) if the integration fails before the envelope is stripped.
    """
    R_core = calculate_R_core_numba(composition, M_core)
    a = (
        ((period * 24.0 * 60.0 * 60.0) ** 2 * G * M_star * M_sun / (4.0 * pi * pi)) ** (1.0 / 3.0)
    ) / AU

    T_eq = _calculate_T_eq(M_star, a)
    c_s_squared = (k_B * T_eq) / (mu * m_H)
    KH0 = initial_KH_timescale * Myr
    M_core_kg = M_core * M_earth
    R_core_meters = R_core * R_earth

    _, _, R_tr_m, _ = _solve_structure(
        T_eq, c_s_squared, KH0, M_core_kg, R_core_meters, initial_X, 0.0, use_Lcore,
    )
    if R_tr_m == 0.0:
        return 0.0, 0.0
    R_tr = R_tr_m / R_earth

    X = initial_X
    t = t_start
    dt = dt_try

    while t < t_stop:
        t_new, X_new, dt_next, status = _step_control(
            t, X, dt, accuracy,
            M_core, M_star, a, R_core, initial_KH_timescale, R_tr,
            use_Lcore, use_aiolos,
            log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
        )
        if status == 1:
            # A nearly stripped envelope is reported as a bare core.
            if X <= STRIPPED_ATMOSPHERE_THRESHOLD:
                return R_core, ATMOSPHERE_FLOOR
            return 0.0, 0.0
        if status == 2:
            return R_core, ATMOSPHERE_FLOOR

        if t_new >= t_stop:
            t_new = t_stop
            # Last partial step, from the current time to t_stop.
            dydx = _dXdt(
                t, X, M_core, M_star, a, R_core, initial_KH_timescale, R_tr,
                use_Lcore, use_aiolos,
                log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
            )
            if dydx == 0.0:
                if X <= STRIPPED_ATMOSPHERE_THRESHOLD:
                    return R_core, ATMOSPHERE_FLOOR
                return 0.0, 0.0
            k1 = dydx * (t_stop - t)
            X_new, _, st = _rkck_step(
                t, X, t_stop - t, k1,
                M_core, M_star, a, R_core, initial_KH_timescale, R_tr,
                use_Lcore, use_aiolos,
                log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
            )
            if st != 0:
                if X <= STRIPPED_ATMOSPHERE_THRESHOLD:
                    return R_core, ATMOSPHERE_FLOOR
                return 0.0, 0.0

        if X_new <= STRIPPED_ATMOSPHERE_THRESHOLD:
            return R_core, ATMOSPHERE_FLOOR

        # Transit radius at the new envelope mass, used as the next radius guess.
        if t_new < initial_KH_timescale:
            KH_seconds = initial_KH_timescale * Myr
        else:
            KH_seconds = t_new * Myr
        T_eq = _calculate_T_eq(M_star, a)
        c_s_squared = (k_B * T_eq) / (mu * m_H)
        _, _, R_tr_m, _ = _solve_structure(
            T_eq, c_s_squared, KH_seconds,
            M_core_kg, R_core_meters, X_new, 0.0, use_Lcore,
        )
        if R_tr_m == 0.0:
            if X_new <= STRIPPED_ATMOSPHERE_THRESHOLD:
                return R_core, ATMOSPHERE_FLOOR
            return 0.0, 0.0
        R_tr = R_tr_m / R_earth

        t = t_new
        dt = dt_next
        X = X_new

    return R_tr, X


def _empty_aiolos_arrays():
    """Placeholder grid used when the energy-limited rate is selected.

    The Aiolos interpolant is not called in that case. The axes only need
    to be long enough for the interpolator to be constructed.
    """
    ax = np.array([0.0, 1.0])
    vals = np.zeros((2, 2, 2, 2))
    return ax, ax, ax, ax, vals


def evolve_planet_fast(
    t_start, t_stop, dt_try, accuracy, initial_X,
    composition, M_core, period, M_star, initial_KH_timescale,
    use_Lcore=True, use_aiolos=False, aiolos_grid=None,
):
    """Evolve one planet and return (transit radius in Earth radii, final X).

    Pass an AiolosMassLossGrid as aiolos_grid to avoid rebuilding the
    tables. With use_aiolos left false, the energy-limited rate is used.
    """
    if use_aiolos:
        if aiolos_grid is None:
            import aiolos_mass_loss
            aiolos_grid = aiolos_mass_loss.get_default_aiolos_grid()
        log_mass_ax = np.asarray(aiolos_grid._log_mass_ax, dtype=np.float64)
        log_teq_ax = np.asarray(aiolos_grid._log_teq_ax, dtype=np.float64)
        log_R_ax = np.asarray(aiolos_grid._log_R_ax, dtype=np.float64)
        log_Luv_ax = np.asarray(aiolos_grid._log_Luv_ax, dtype=np.float64)
        grid_vals = np.asarray(aiolos_grid._rgi.values, dtype=np.float64)
    else:
        log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals = _empty_aiolos_arrays()

    return evolve_planet_numba(
        float(t_start), float(t_stop), float(dt_try), float(accuracy),
        float(initial_X), float(composition), float(M_core), float(period),
        float(M_star), float(initial_KH_timescale),
        bool(use_Lcore), bool(use_aiolos),
        log_mass_ax, log_teq_ax, log_R_ax, log_Luv_ax, grid_vals,
    )
