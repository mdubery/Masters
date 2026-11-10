"""Evolution of a planet's hydrogen envelope under photoevaporation.

The evolved quantity is the envelope mass fraction X = M_envelope / M_core.
It obeys

    dX/dt = -X / t_X

where t_X = M_envelope / M_dot is the mass-loss timescale. The mass-loss
rate is either the energy-limited rate or an interpolation of the Aiolos
hydrodynamic grid. The planet's transit radius is recomputed from the
envelope structure at each step.
"""

import numpy as np
from numpy import log10

from constants import *
import planet_structure
import RKF45
import aiolos_mass_loss

# Below this envelope fraction the planet is treated as a bare core.
ATMOSPHERE_FLOOR = 1e-4
STRIPPED_ATMOSPHERE_THRESHOLD = 5e-4


def R_imf(imf, M_core):
    """Core radius in Earth radii for an ice-rock mixture (Fortney et al. 2007).

    imf is the ice mass fraction, between 0 and 1. M_core is in Earth masses.
    """

    _R1 = (0.0912*imf + 0.1603) * (log10(M_core)*log10(M_core))
    _R2 = (0.3330*imf + 0.7387) *  log10(M_core)
    _R3 = (0.4639*imf + 1.1193)

    return _R1 + _R2 + _R3

def R_rmf(rmf, M_core):
    """Core radius in Earth radii for a rock-iron mixture (Fortney et al. 2007).

    rmf is the rock mass fraction, between 0 and 1. M_core is in Earth masses.
    """

    _R1 = (0.0592*rmf + 0.0975) * (log10(M_core)*log10(M_core))
    _R2 = (0.2337*rmf + 0.4938) *  log10(M_core)
    _R3 = (0.3102*rmf + 0.7932)

    return _R1 + _R2 + _R3


def calculate_R_core(composition, M_core):
    """Core radius in Earth radii.

    composition > 0 is the iron mass fraction of a rock-iron core, so the
    rock fraction is 1 - composition. composition <= 0 selects an ice-rock
    core whose ice fraction is -composition. M_core is in Earth masses.
    """
    if composition <= 0.0:
        imf = -1.0 * composition
        R_core = R_imf(imf, M_core)
    if composition > 0.0:
        rmf = 1.0 - composition
        R_core = R_rmf(rmf, M_core)

    return R_core


def calculate_tX(t, X, M_core, M_star, smet, a, R_core,
                 initial_KH_timescale, R_guess, Lbol_interp,
                 use_Lcore, use_aiolos=False,
                 aiolos_grid=None, return_LxLbol=False):
    """Mass-loss timescale t_X = M_envelope / M_dot, in megayears.

    t is the age in Myr, X = M_envelope / M_core, and a is the orbital
    separation in AU. Masses and radii are in Earth units. Returns 0 if the
    envelope structure cannot be solved.

    use_aiolos selects the Aiolos hydrodynamic rate. Otherwise the rate is
    the energy-limited rate of Owen & Wu (2017). With return_LxLbol, and a
    stellar-evolution luminosity supplied, the return is L_X / L_bol instead.
    """
    # convert to SI units
    M_core_kg = M_core * M_earth
    a_meters = a * AU
    t_seconds = t * Myr
    R_core_meters = R_core * R_earth

    M_env_kg = X * M_core_kg
    # Stars below about a solar mass stay X-ray saturated for longer.
    t_sat = 100 * Myr * (M_star**(-1.0))
    R_rcb, Rho_rcb, T_eq, P_rcb, R_transit, R_xuv = planet_structure.calculate_R_transit(t, M_star, smet, a, M_core, R_core, X, initial_KH_timescale, R_guess, Lbol_interp, use_Lcore, return_extra=True)

    if R_rcb == 0.0:
        return 0.0

        

    # Stellar-evolution tracks, if supplied. Otherwise a mass-luminosity relation.
    if Lbol_interp:

        LxLbol_sat = 10**(-3.50) * (M_star**(-0.5))
        if t_seconds < t_sat:
            LxLbol = LxLbol_sat
        else:
            LxLbol = LxLbol_sat * (t_seconds/t_sat)**(-1.3)

        if t > 5000:
            Lbol = 10**Lbol_interp([M_star, smet, np.log10(5000 * 1e6)])[0]
        else:
            Lbol = 10**Lbol_interp([M_star, smet, np.log10(t * 1e6)])[0]
        Lx = L_sun * LxLbol * Lbol

    else:
        # Saturated X-ray luminosity, then a t^-1.3 decline after t_sat.
        L_sat = 10**(-3.50) * L_sun * (M_star**(0.5))
        if t_seconds < t_sat:
            Lx = L_sat
        else:
            Lx = L_sat * (t_seconds/t_sat)**(-1.3)
    
    if return_LxLbol:
        return Lx/(Lbol)

    # Efficiency falls with escape speed. 23 km/s is the reference speed.
    escape_velocity = np.sqrt(2*G*M_core_kg / R_core_meters) * 0.001
    eta = eta_0 * (escape_velocity / 23)**(-0.42)

    # Energy-limited rate. R_transit is the optical photosphere.
    M_env_dot_PE = eta * R_transit**3 * Lx / (4 * a_meters * a_meters * G * M_core_kg)

    if use_aiolos:
        if aiolos_grid is None:
            aiolos_grid = aiolos_mass_loss.get_default_aiolos_grid()

        T_eq = planet_structure.calculate_T_eq(M_star, smet, a, t, Lbol_interp)
        M_planet_earth = M_core * (1.0 + X)
        R_transit_cm = R_transit * 1e2   # metres to centimetres
        L_uv_erg_s = Lx * 1e7            # watts to erg/s
        M_env_dot_AIOLOS = aiolos_grid.evaluate(
            planet_mass_earth=M_planet_earth,
            teq_k=T_eq,
            transit_radius_cm=R_transit_cm,
            luv_erg_s=L_uv_erg_s,
        ) / 1000.0  # g/s to kg/s

    if use_aiolos:
        M_env_dot = M_env_dot_AIOLOS
    else:
        M_env_dot = M_env_dot_PE
    
    tX = (M_env_kg / M_env_dot) / Myr

    return tX

def dXdt_ODE(t, X, parameters):
    """Right-hand side of the evolution equation, dX/dt = -X / t_X.

    parameters is the list assembled in RK45_driver.
    """
    (
        M_core,
        M_star,
        smet,
        a,
        R_core,
        initial_KH_timescale,
        R_guess,
        Lbol_interp,
        use_Lcore,
        use_aiolos,
        aiolos_grid,
    ) = parameters
    tX = calculate_tX(
        t,
        X,
        M_core,
        M_star,
        smet,
        a,
        R_core,
        initial_KH_timescale,
        R_guess,
        Lbol_interp,
        use_Lcore,
        use_aiolos=use_aiolos,
        aiolos_grid=aiolos_grid,
    )

    if tX == 0.0:
        return 0.0

    dXdt = - X / tX

    return dXdt


def RK45_driver(t_start, t_stop, dt_try, accuracy, initial_X, composition,
                M_core, period, M_star, smet, initial_KH_timescale,
                return_history=False, MIST_interp=False,
                use_Lcore=True, use_aiolos=False, aiolos_grid=None):
    """Integrate one planet from t_start to t_stop.

    Times are in megayears. initial_X is the envelope mass fraction.
    composition is passed to calculate_R_core. M_core is in Earth masses,
    period in days, and M_star in solar masses. smet is the stellar
    metallicity. It is used only when MIST_interp is a function giving the
    stellar luminosity from evolutionary tracks; leave MIST_interp as False
    to use a mass-luminosity relation L proportional to M^4.5.
    initial_KH_timescale is the early envelope cooling time in megayears.

    dt_try is the first trial timestep and accuracy is the relative
    tolerance of the adaptive stepper. use_Lcore includes the core
    luminosity in the envelope structure. use_aiolos takes mass-loss rates
    from the Aiolos grid instead of the energy-limited formula.

    Without return_history the result is (final transit radius in Earth
    radii, final X). A failed integration returns (0, 0). An envelope that
    drops below the stripping threshold returns the bare-core radius and
    X = ATMOSPHERE_FLOOR. With return_history the result is three lists:
    time, X, and transit radius.
    """

    # The compiled integrator does not store the time series and does not
    # accept stellar-evolution tracks.
    if (not return_history) and (not MIST_interp):
        import fast_evolver
        return fast_evolver.evolve_planet_fast(
            t_start=t_start,
            t_stop=t_stop,
            dt_try=dt_try,
            accuracy=accuracy,
            initial_X=initial_X,
            composition=composition,
            M_core=M_core,
            period=period,
            M_star=M_star,
            initial_KH_timescale=initial_KH_timescale,
            use_Lcore=use_Lcore,
            use_aiolos=use_aiolos,
            aiolos_grid=aiolos_grid,
        )

    if return_history:
        t_array = []
        X_array = []
        Rtr_array = []

    R_core = calculate_R_core(composition, M_core)
    # Kepler's law: orbital period to separation in AU.
    a = (((period * 24 * 60 * 60)**2 * G * M_star * M_sun / (4 * pi * pi))**(1/3)) / AU
    R_tr_init = planet_structure.calculate_R_transit(t_start, M_star, smet, a, M_core, R_core, initial_X, initial_KH_timescale, 0.0, MIST_interp, use_Lcore) / R_earth

    if return_history:
        t_array.append(t_start)
        X_array.append(initial_X)
        Rtr_array.append(R_tr_init)

    X = initial_X
    t = t_start
    R_tr = R_tr_init
    dt = dt_try

    def _return_stripped(t_stripped=None):
        """Record a stripped planet as a bare core for the rest of the run."""
        if return_history:
            if t_stripped is None:
                t_stripped = t_stop
            t_array.append(t_stripped)
            X_array.append(ATMOSPHERE_FLOOR)
            Rtr_array.append(R_core)
            if t_stripped < t_stop:
                t_array.append(t_stop)
                X_array.append(ATMOSPHERE_FLOOR)
                Rtr_array.append(R_core)
            return t_array, X_array, Rtr_array
        return R_core, ATMOSPHERE_FLOOR

    while t < t_stop:
        [t_new, X_new, dt_next] = RKF45.step_control(
            t,
            X,
            dt,
            dXdt_ODE,
            accuracy,
            parameters=[
                M_core,
                M_star,
                smet,
                a,
                R_core,
                initial_KH_timescale,
                R_tr,
                MIST_interp,
                use_Lcore,
                use_aiolos,
                aiolos_grid,
            ],
            terminal_min_value=ATMOSPHERE_FLOOR,
        )

        if [t_new, X_new, dt_next] == [0.0, 0.0, 0.0]:
            if X <= STRIPPED_ATMOSPHERE_THRESHOLD:
                return _return_stripped(t)
            return 0.0, 0.0

        if t_new >= t_stop:
            t_new = t_stop
            X_new, _ = RKF45.RKCashKarp_Step45(
                t,
                X,
                dXdt_ODE,
                t_stop-t,
                [
                    M_core,
                    M_star,
                    smet,
                    a,
                    R_core,
                    initial_KH_timescale,
                    R_tr,
                    MIST_interp,
                    use_Lcore,
                    use_aiolos,
                    aiolos_grid,
                ],
            )

        # A very thin envelope is not given to the structure solver.
        if X_new is None or X_new <= STRIPPED_ATMOSPHERE_THRESHOLD:
            return _return_stripped(t_new)

        # Transit radius at the new envelope mass, used as the next radius guess.
        _rcb_result = planet_structure.calculate_R_transit(
            t_new, M_star, smet, a, M_core, R_core, X_new,
            initial_KH_timescale, 0.0, MIST_interp, use_Lcore, return_extra=True
        )
        if _rcb_result[0] == 0.0:
            if X_new <= STRIPPED_ATMOSPHERE_THRESHOLD:
                return _return_stripped(t_new)
            return 0.0, 0.0

        R_rcb, Rho_rcb, T_eq, P_rcb, R_tr_new, R_xuv_new = _rcb_result
        R_tr_new = R_tr_new / R_earth

        if return_history:
            t_array.append(t_new)
            X_array.append(X_new)
            Rtr_array.append(R_tr_new)

        t = t_new
        dt = dt_next
        X = X_new
        R_tr = R_tr_new

    if return_history:
        return t_array, X_array, Rtr_array
    else:
        return R_tr, X


