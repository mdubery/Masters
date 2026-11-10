"""Analytic structure of a hydrogen envelope on a rocky core.

The model is the convective-envelope solution of Owen & Wu (2017). Given the
envelope mass fraction, it returns the radius of the radiative-convective
boundary and the optical transit radius.
"""

import numpy as np
from scipy.optimize import brentq
from constants import *


def I2(dR_Rc):
    """Dimensionless mass integral of the convective envelope.

    dR_Rc is (R_rcb / R_core) - 1. The value is a fit to the integral in
    Owen & Wu (2017).
    """
    log_dR_Rc = np.log10(dR_Rc)
    log_I2 = np.log(1.2598238012325735 + np.exp(-2.4535812359864062*1.0798633578446974*log_dR_Rc)) / -1.0798633578446974 - 4.8814961274562990e-01
    I2 = 10**log_I2
    return I2


def I2_I1(dR_Rc):
    """Ratio of the two convective-envelope integrals in Owen & Wu (2017).

    dR_Rc is (R_rcb / R_core) - 1.
    """
    log_dR_Rc = np.log10(dR_Rc)
    log_I2_I1 = -7.6521840215423576e-01 / (1.0 + np.exp(-(log_dR_Rc-5.7429970641208375e-02)/6.4338705851296174e-01))**1.8214254374336605E+00
    I2_I1 = 10**log_I2_I1
    return I2_I1

def calculate_T_eq(M_star, smet, a, t, Lbol_interp):
    """Equilibrium temperature in kelvin.

    M_star is in solar masses and a is the orbital separation in AU.
    With Lbol_interp left as False, the star is given a main-sequence
    mass-luminosity relation L proportional to M^4.5. If Lbol_interp is a
    callable, it supplies log10(L/L_sun) from stellar-evolution tracks at
    (M_star, metallicity, log10(age in years)). Ages above 5000 Myr are
    held at 5000 Myr so the star is not evolved off the main sequence.
    """

    if Lbol_interp:
        if t > 5000: # don't consider stars leaving the main-sequence
            Lbol = 10**Lbol_interp([M_star, smet, np.log10(5000 * 1e6)])[0] * L_sun
        else:
            Lbol = 10**Lbol_interp([M_star, smet, np.log10(t * 1e6)])[0] * L_sun

        a_meters = a * AU
        T_eq = (Lbol / (16 * stefan * pi * a_meters* a_meters))**0.25

    else:

        T_eq = T_eq_earth * (1/a)**0.5 * (M_star)**(0.25 * 4.5)


    return T_eq


def calculate_sound_speed_squared(T_eq):
    """Isothermal sound speed squared, c_s^2 = kT / (mu m_H), in m^2 s^-2."""

    c_s_squared = (k_B * T_eq) / (mu * m_H)
    return c_s_squared

def R_rcb_equation(R_rcb, T_eq, c_s_squared, KH_timescale_seconds, M_core_kg, R_core_meters, X, use_Lcore):
    """Residual of the radiative-convective boundary equation.

    A root in R_rcb (metres) is the boundary radius for this envelope.
    X <= 0 means the envelope has gone, so the residual is NaN.
    use_Lcore includes the luminosity of the cooling core; without it the
    only luminosity is the cooling of the envelope itself.
    """
    if not np.isfinite(X) or X <= 0.0:
        return np.nan

    # psi = 1 includes the core luminosity, psi = 0 leaves it out.
    if use_Lcore:
        psi = 1.0
    else:
        psi=0.0

    # Terms that do not depend on R_rcb.
    c1 = (4 * pi * mu * m_H / (M_core_kg * k_B))
    c2 = ((Delta_ab * G * M_core_kg / c_s_squared)**(1/(gamma-1)))
    c3 = ((1088 * pi * stefan * (T_eq**(3-alpha-beta)) * KH_timescale_seconds / (3 * kappa_0 * M_core_kg * (17*X+psi)))**(1/(alpha+1)))

    c = c1 * c2 * c3

    equation = c * (R_rcb**3) * I2((R_rcb/R_core_meters)-1) * ((1/R_rcb)**(1/(gamma-1))) * ((I2_I1((R_rcb/R_core_meters)-1) * R_rcb)**(1/(alpha+1))) - X

    return equation



def solve_Rho_rcb_and_R_rcb(T_eq, c_s_squared, KH_timescale_seconds, M_core_kg, R_core_meters, X, R_guess, use_Lcore):
    """Solve for the radiative-convective boundary.

    Returns (R_rcb, Rho_rcb) in metres and kg m^-3. Returns (0, 0) if the
    bracket contains no root. R_guess is a previous transit radius in Earth
    radii, used only to choose the first upper bound; a non-positive guess
    falls back to 500 Earth radii.
    """


    brentq_args = (T_eq, c_s_squared, KH_timescale_seconds, M_core_kg, R_core_meters, X, use_Lcore)
    lo = R_core_meters * 1.0001

    max_hi = 500.0 * R_earth
    # A missing or negative guess must not place the upper bound inside the core.
    if R_guess <= 0.0:
        hi = max_hi
    else:
        hi = min(max_hi, R_earth * (1.0 + R_guess))

    # Widen the bracket until the residual changes sign, or the cap is reached.
    f_lo = R_rcb_equation(lo, *brentq_args)
    f_hi = R_rcb_equation(hi, *brentq_args)
    while (
        np.isfinite(f_lo)
        and np.isfinite(f_hi)
        and np.sign(f_lo) == np.sign(f_hi)
        and hi < max_hi
    ):
        hi = min(max_hi, 2.0 * hi)
        f_hi = R_rcb_equation(hi, *brentq_args)

    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or np.sign(f_lo) == np.sign(f_hi):
        return 0.0, 0.0

    try:
        R_rcb = brentq(R_rcb_equation, lo, hi, args=brentq_args, disp=False)
    except ValueError:
        return 0.0, 0.0



    Rho_rcb_1 = (mu * m_H / k_B)
    if use_Lcore:
        psi = 1.0
    else:
        psi = 0.0
    Rho_rcb_2 = ((I2_I1((R_rcb/R_core_meters)-1) * 1088 * pi * stefan * (T_eq**(3-alpha-beta)) * KH_timescale_seconds * R_rcb / (3 * kappa_0 * M_core_kg * (17*X+psi)))**(1/(alpha+1)))
    Rho_rcb = Rho_rcb_1 * Rho_rcb_2


    return R_rcb, Rho_rcb



def calculate_R_transit(t, M_star, smet, a, M_core, R_core, X, KH_timescale_cutoff, R_guess, Lbol_interp, use_Lcore, return_extra=False):
    """Optical transit radius in metres.

    t is the age in Myr. M_star is in solar masses, a in AU, M_core and
    R_core in Earth units, and X = M_envelope / M_core. The cooling time is
    KH_timescale_cutoff (Myr) while the planet is younger than that, and
    equals the age afterwards.

    Returns 0 if the boundary radius cannot be found. With return_extra,
    the return is (R_rcb, Rho_rcb, T_eq, P_rcb, R_transit, R_xuv).
    """
    T_eq = calculate_T_eq(M_star, smet, a, t, Lbol_interp)
    c_s_squared = calculate_sound_speed_squared(T_eq)

    if t < KH_timescale_cutoff:
        KH_timescale_seconds = KH_timescale_cutoff*Myr
    else:
        KH_timescale_seconds = t*Myr

    M_core_kg = M_core * M_earth
    R_core_meters = R_core * R_earth

    R_rcb, Rho_rcb = solve_Rho_rcb_and_R_rcb(T_eq, c_s_squared, KH_timescale_seconds, M_core_kg, R_core_meters, X, R_guess, use_Lcore)
    P_rcb = c_s_squared * Rho_rcb

    if (R_rcb, Rho_rcb) == (0.0, 0.0):
        if return_extra:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        else:
            return 0.0

    Rbondi = G * M_core_kg * (1 + X) / (2 * c_s_squared)

    # Attach an isothermal layer out to the transit (20 mbar) and XUV (1 nbar) radii.
    R_transit = _isothermal_attached_radius(R_rcb, P_TRANSIT, P_rcb, Rbondi)
    R_xuv = _isothermal_attached_radius(R_rcb, P_XUV, P_rcb, Rbondi)

    if return_extra:
        return R_rcb, Rho_rcb, T_eq, P_rcb, R_transit, R_xuv
    else:
        return R_transit


def _isothermal_attached_radius(R_rcb, P_level, P_rcb, Rbondi):
    """Radius of an isothermal layer sitting on the radiative-convective boundary.

    R = R_rcb / (1 + 0.5 R_rcb ln(P_level / P_rcb) / R_bondi).

    Both pressures are in pascals. For a hot or low-mass planet the formal
    result can fall outside [R_rcb, R_bondi]; it is clipped back into that range.
    """
    if R_rcb <= 0.0 or Rbondi <= 0.0 or P_rcb <= 0.0 or P_level <= 0.0:
        return R_rcb

    denom = 1.0 + 0.5 * R_rcb * np.log(P_level / P_rcb) / Rbondi
    if (not np.isfinite(denom)) or denom <= 0.0:
        # The isothermal layer is unbound. Use the Bondi radius, but stay
        # outside the radiative-convective boundary.
        return max(R_rcb, Rbondi)

    R = R_rcb / denom
    if (not np.isfinite(R)) or R <= 0.0:
        return R_rcb

    # The photosphere lies between the boundary and the Bondi radius.
    if R < R_rcb:
        return R_rcb
    if R > Rbondi:
        return Rbondi
    return R


