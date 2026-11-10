"""Adaptive Cash-Karp integrator for the envelope-mass equation.

The evolution advances the envelope mass fraction X. Each accepted step
returns a fifth-order update and an error estimate from the difference
between that update and an embedded fourth-order update.
"""

import numpy as np


def RKCashKarp_Step45(t, var_i, dydx_functions, dt, parameters, k1=None):
    """Take one Cash-Karp step.

    Returns (fifth-order solution, error estimate). A failed atmosphere
    structure calculation returns ("BRENTQ ERROR", "BRENTQ ERROR"). A step
    that would make X negative returns (None, None), which tells the
    controller to try a shorter timestep.

    k1, if given, is dydx * dt already evaluated at the start of the step.
    """

    # k1 -------------------------------------------------------------------
    if k1 is None:
        dydx = dydx_functions(t, var_i, parameters)
        if dydx == 0.0:
            return ("BRENTQ ERROR", "BRENTQ ERROR")
        k1 = dydx * dt
    # (if k1 was provided it is already dydx*dt and known non-zero)

    var1 = var_i + 0.2 * k1
    if var1 < 0:
        return (None, None)

    # k2 -------------------------------------------------------------------
    k2 = dydx_functions(t + 0.2 * dt, var1, parameters)
    if k2 == 0.0:
        return ("BRENTQ ERROR", "BRENTQ ERROR")
    k2 *= dt

    var2 = var_i + (3/40)*k1 + (9/40)*k2
    if var2 < 0:
        return (None, None)

    # k3 -------------------------------------------------------------------
    k3 = dydx_functions(t + 0.3 * dt, var2, parameters)
    if k3 == 0.0:
        return ("BRENTQ ERROR", "BRENTQ ERROR")
    k3 *= dt

    var3 = var_i + (3/10)*k1 - (9/10)*k2 + (6/5)*k3
    if var3 < 0:
        return (None, None)

    # k4 -------------------------------------------------------------------
    k4 = dydx_functions(t + 0.6 * dt, var3, parameters)
    if k4 == 0.0:
        return ("BRENTQ ERROR", "BRENTQ ERROR")
    k4 *= dt

    var4 = var_i - (11/54)*k1 + (5/2)*k2 - (70/27)*k3 + (35/27)*k4
    if var4 < 0:
        return (None, None)

    # k5 -------------------------------------------------------------------
    k5 = dydx_functions(t + dt, var4, parameters)
    if k5 == 0.0:
        return ("BRENTQ ERROR", "BRENTQ ERROR")
    k5 *= dt

    var5 = (var_i + (1631/55296)*k1 + (175/512)*k2
            + (575/13824)*k3 + (44275/110592)*k4 + (253/4096)*k5)
    if var5 < 0:
        return (None, None)

    # k6 -------------------------------------------------------------------
    k6 = dydx_functions(t + (7/8)*dt, var5, parameters)
    if k6 == 0.0:
        return ("BRENTQ ERROR", "BRENTQ ERROR")
    k6 *= dt

    # RK5 and RK4 solutions -----------------------------------------------
    RK5_sol = var_i + (37/378)*k1 + (250/621)*k3 + (125/594)*k4 + (512/1771)*k6
    RK4_sol = (var_i + (2825/27648)*k1 + (18575/48384)*k3
               + (13525/55296)*k4 + (277/14336)*k5 + 0.25*k6)

    # The envelope mass fraction cannot be negative.
    if RK5_sol < 0.0:
        return (None, None)

    err = RK5_sol - RK4_sol
    return (RK5_sol, err)


def RKFehlberg12(t, var_i, dydx_functions, dt, parameters):
    """Take one Fehlberg RK2/1 step.

    Same return convention as RKCashKarp_Step45. The planet evolution uses
    the Cash-Karp step above.
    """

    k1 = dt * dydx_functions(t, var_i, parameters)
    var1 = var_i + 0.5 * k1
    if var1 < 0:
        return (None, None)

    k2 = dt * dydx_functions(t + 0.5 * dt, var1, parameters)
    var2 = var_i + (1/256)*k1 + (255/256)*k2
    if var2 < 0:
        return (None, None)

    k3 = dt * dydx_functions(t + dt, var2, parameters)

    RK1_sol = var_i + (1/256)*k1 + (255/256)*k2
    RK2_sol = var_i + (1/512)*k1 + (255/256)*k2 + (1/512)*k3
    err = RK2_sol - RK1_sol
    return (RK2_sol, err)


def step_control(t, var_i, dt_try, dydx_functions, accuracy, parameters,
                 terminal_min_value=None):
    """Advance one accepted step, choosing the timestep from the error estimate.

    Returns [t_new, X_new, dt_next]. The derivative is evaluated once and
    reused as k1 inside the Cash-Karp step.

    If the timestep underflows while X is still decreasing, and
    terminal_min_value is set, the envelope is reported as that floor value.
    A structure failure returns [0, 0, 0].
    """

    def _underflow_result():
        if terminal_min_value is not None and dydx < 0.0:
            return [t, terminal_min_value, 0.0]
        return [0.0, 0.0, 0.0]

    # Evaluate derivative once; reuse as k1 inside the RK step.
    dydx = dydx_functions(t, var_i, parameters)
    if dydx == 0.0:
        return [0.0, 0.0, 0.0]

    yscal = abs(var_i) + abs(dydx * dt_try) + 1e-3
    dt = dt_try

    while True:
        # Pass the pre-computed k1 to avoid a redundant function call.
        k1 = dydx * dt
        (var_new, var_err) = RKCashKarp_Step45(
            t, var_i, dydx_functions, dt, parameters, k1=k1
        )

        if (var_new, var_err) == (None, None):
            dt *= 0.01
            if t + dt == t:
                return _underflow_result()
            continue

        if (var_new, var_err) == ("BRENTQ ERROR", "BRENTQ ERROR"):
            return [0.0, 0.0, 0.0]

        err_max = abs(var_err / yscal) / accuracy

        if err_max <= 1.0:
            break

        # Step failed: shrink dt but not by more than factor 10.
        dt_new = 0.9 * dt / (err_max ** 0.25)
        dt = max(dt_new, 0.1 * dt)

        if t + dt == t:
            return _underflow_result()

    # Propose next step size.
    if err_max < 2e-4:
        dt_next = 5.0 * dt
    else:
        dt_next = 0.9 * (dt / err_max) ** 0.2

    return [t + dt, var_new, dt_next]
