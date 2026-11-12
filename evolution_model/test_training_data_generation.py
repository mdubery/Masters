
import matplotlib.pyplot as plt
import numpy as np
import csv
from time import time
start_time = time()



from pathlib import Path
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



def run(P, M_s):
    t_myr, envelope_fraction, radius_earth = mass_fraction_evolver.RK45_driver(
        t_start=t_start,
        t_stop=t_stop,
        dt_try=0.001,              # first trial timestep, Myr
        accuracy=1e-5,             # relative tolerance of the adaptive stepper
        initial_X=initial_X,
        composition=composition,
        M_core=M_core,
        period=P,
        M_star=M_s,
        smet=0.0,                  # stellar metallicity; unused with the default luminosity
        initial_KH_timescale=initial_KH_timescale,
        return_history=True,       # return the time series, not only the final state
        use_Lcore=True,            # include the luminosity of the cooling core
        use_aiolos=True,           # mass-loss rates from the Aiolos hydrodynamic grid
    )
    return t_myr, envelope_fraction, radius_earth


output_file = Path(__file__).parent / "training_data.csv"
with output_file.open(mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Period (days)", "Stellar Mass (M_sun)", "Final Envelope Fraction", "Final Radius (R_earth)"])
    
    # Loop over a range of periods and stellar masses
    for P in np.linspace(1, 100, 10):  # Example: 10 periods from 1 to 100 days
        for M_s in np.linspace(0.5, 2.0, 10):  # Example: 10 stellar masses from 0.5 to 2.0 M_sun
            t_myr, envelope_fraction, radius_earth = run(P, M_s)
            writer.writerow([P, M_s, envelope_fraction[-1], radius_earth[-1]])
            f.flush()
            print("done")