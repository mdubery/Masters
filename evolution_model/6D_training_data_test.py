
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


from scipy.stats import qmc


#latin hypercube sampling to generate training data
def LHC_sample_parameters(n_samples) -> np.ndarray:
    seed = 42 
    """Generate space-filling (period_days, stellar_mass_msun) pairs."""
    sampler = qmc.LatinHypercube(d=6, seed=seed)
    unit_samples = sampler.random(n=n_samples)

    # Sample log10(P) uniformly between 0 (1 day) and 2 (100 days),
    # and M_star uniformly between 0.5 and 2.0 M_sun
    # Metalicity (-1 , 0.5)
    # M_core (0.6, 20)
    # rho = (-1, 1)
    # logX_int (-4, 0)

    lower_bounds = [0.0, 0.5, -1,  0.6, -1, -4]
    upper_bounds = [2.0, 2.0, 0.5, 20,   1,  0]
    scaled = qmc.scale(unit_samples, lower_bounds, upper_bounds)

    periods = 10.0 ** scaled[:, 0]
    stellar_masses = scaled[:, 1]
    metalicity = scaled[:, 2]
    M_core = scaled[:, 3]
    rho = scaled[:, 4]
    logX_int = 10**scaled[:, 5]

    return np.column_stack((periods, stellar_masses, metalicity, M_core, rho, logX_int))

N = 10000
sample = LHC_sample_parameters(N)

def run(P, M_s, Z, M_c, rho, X_int):
    t_myr, envelope_fraction, radius_earth = mass_fraction_evolver.RK45_driver(
        t_start=t_start,
        t_stop=t_stop,
        dt_try=0.001,              # first trial timestep, Myr
        accuracy=1e-5,             # relative tolerance of the adaptive stepper
        initial_X= X_int,
        composition=rho,
        M_core=M_c,
        period=P,
        M_star=M_s,
        smet=Z,                  # stellar metallicity; unused with the default luminosity
        initial_KH_timescale=initial_KH_timescale,
        return_history=True,       # return the time series, not only the final state
        use_Lcore=True,            # include the luminosity of the cooling core
        use_aiolos=True,           # mass-loss rates from the Aiolos hydrodynamic grid
    )
    return t_myr, envelope_fraction, radius_earth


output_file = Path(__file__).parent / "training_data_6D.csv"
with output_file.open(mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Period (days)", "Stellar Mass (M_sun)", "Metalicity","M_core", "rho", "Log x int",  "Final Envelope Fraction", "Final Radius (R_earth)"])
    
    
    for i ,(P, M_s, Z, M_c, rho, X_int) in enumerate(sample, start =1):
        t_myr, envelope_fraction, radius_earth = run(P, M_s, Z, M_c, rho, X_int)
        writer.writerow([P, M_s, Z, M_c, rho, X_int, envelope_fraction[-1], radius_earth[-1]])
        f.flush()
        if i % (N//100) == 0:
            print(int(100*i/len(sample)), "%")

end_time = time()

print(end_time - start_time)
