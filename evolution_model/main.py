"""Evolve one planet and plot its transit radius and envelope mass fraction.

Run from this directory:

    python main.py

Time is in megayears. The envelope mass fraction is X = M_envelope / M_core.
The radius is the optical transit radius, in Earth radii.
"""

import matplotlib.pyplot as plt

import mass_fraction_evolver


# Change these to evolve a different planet.
t_start = 10.0                 # age at the start of the integration, Myr
t_stop = 5000.0                # age at the end, Myr
initial_X = 0.03               # initial envelope mass fraction
composition = 0.33             # core iron mass fraction (Fortney et al. 2007)
M_core = 10.0                  # core mass, Earth masses
period = 10.0                  # orbital period, days
M_star = 1.0                   # stellar mass, solar masses
initial_KH_timescale = 100.0  # early cooling time of the envelope, Myr

t_myr, envelope_fraction, radius_earth = mass_fraction_evolver.RK45_driver(
    t_start=t_start,
    t_stop=t_stop,
    dt_try=0.001,              # first trial timestep, Myr
    accuracy=1e-5,             # relative tolerance of the adaptive stepper
    initial_X=initial_X,
    composition=composition,
    M_core=M_core,
    period=period,
    M_star=M_star,
    smet=0.0,                  # stellar metallicity; unused with the default luminosity
    initial_KH_timescale=initial_KH_timescale,
    return_history=True,       # return the time series, not only the final state
    use_Lcore=True,            # include the luminosity of the cooling core
    use_aiolos=True,           # mass-loss rates from the Aiolos hydrodynamic grid
)

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6.0, 6.5))
axes[0].plot(t_myr, radius_earth, color="0.15")
axes[0].set_ylabel(r"Transit radius [$R_\oplus$]")
axes[1].plot(t_myr, envelope_fraction, color="0.15")
axes[1].set_ylabel("Envelope mass fraction")
axes[1].set_xlabel("Time [Myr]")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
fig.tight_layout()
plt.show()
