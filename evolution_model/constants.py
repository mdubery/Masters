"""Physical constants and parameters for the planet-evolution model.

SI units unless a comment says otherwise.
"""

import numpy as np

pi = np.pi
G = 6.674e-11            # gravitational constant, m^3 kg^-1 s^-2
stefan = 5.67e-8         # Stefan-Boltzmann constant, W m^-2 K^-4
k_B = 1.381e-23          # Boltzmann constant, J K^-1
m_H = 1.67e-27           # hydrogen atom mass, kg

# Envelope gas. mu is the mean molecular weight; gamma is the adiabatic index.
mu = 2.35
gamma = 5 / 3
Delta_ab = (gamma - 1) / gamma

# Opacity kappa = kappa_0 * rho^alpha * T^beta (Owen & Wu 2017).
kappa_0 = 2.294e-8
alpha = 0.68
beta = 0.45

# Reference masses, luminosities and lengths.
M_sun = 1.989e30         # kg
L_sun = 3.828e26         # W
R_sun = 6.957e8          # m
AU = 1.496e11            # m
M_earth = 5.972e24       # kg
R_earth = 6.378e6        # m

day = 24 * 60 * 60       # seconds
Myr = 1e6 * 365 * day    # seconds

# Flux and equilibrium temperature of an Earth-like orbit around the Sun.
S_earth = L_sun / (4 * pi * AU * AU)
T_eq_earth = (L_sun / (16 * stefan * pi * AU * AU)) ** 0.25

# Energy-limited mass-loss efficiency at an escape speed of 23 km/s.
eta_0 = 0.1

# Pressures used to place the observable radii. P_rcb is in pascals, so these are too.
_BAR_TO_PA = 1.0e5
P_TRANSIT = 20.0e-3 * _BAR_TO_PA   # 20 mbar: optical transit photosphere
P_XUV = 1.0e-9 * _BAR_TO_PA        # 1 nbar: X-ray/ultraviolet photosphere
