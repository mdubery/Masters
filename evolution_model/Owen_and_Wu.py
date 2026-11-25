import joblib
import mass_fraction_evolver
import matplotlib.pyplot as plt
import numpy as np

# Change these to evolve a different planet.
t_start = 1                # age at the start of the integration, Myr
t_stop = 5000              # age at the end, Myr



initial_KH_timescale = 100.0  # early cooling time of the envelope, Myr



def run_sim(P, M_s, Z, M_c, rho, X_int):
    t_myr, envelope_fraction, radius_earth = mass_fraction_evolver.RK45_driver(
        t_start=t_start,
        t_stop=t_stop,
        dt_try=1e-6,              # first trial timestep, Myr
        accuracy=1e-10,             # relative tolerance of the adaptive stepper
        initial_X= X_int,           # initial envelope mass fraction
        composition=rho,            # core iron mass fraction (Fortney et al. 2007)
        M_core=M_c,                 # core mass, Earth masses
        period=P,
        M_star=M_s,                 # stellar mass, solar masses
        smet=Z,                  # stellar metallicity; unused with the default luminosity
        initial_KH_timescale=initial_KH_timescale,
        return_history=True,       # return the time series, not only the final state
        use_Lcore=True,            # include the luminosity of the cooling core
        use_aiolos=True,           # mass-loss rates from the Aiolos hydrodynamic grid
    )
    return t_myr, envelope_fraction, radius_earth



P = 20
M_s =1
Z = 0.02
M_c = 10
rho = 0.33
X_int = np.logspace(-3.5, 0, 20)


for X_i in X_int:
    t, X, Y  = run_sim(P, M_s, Z, M_c, rho, X_i)
    t = np.array(t) * 1e6
    plt.plot(t, X)
    



#check against the MLPR emulator

bundle = joblib.load("evolution_model/two_stage_emulator.joblib")
clf_scaler = bundle["clf_scaler"]
clf = bundle["clf"]
reg_scaler = bundle["reg_scaler"]
reg = bundle["reg"]

def MLPR_predict_envelope(period, mass, metallicity, m_core, rho, x_int):
    X_new = np.column_stack((
        np.log10(period), mass, metallicity, m_core, rho, np.log10(x_int)
    ))
    is_retained = clf.predict(clf_scaler.transform(X_new)) == 1
    
    pred = np.zeros(len(X_new))
    if np.any(is_retained):
        pred[is_retained] = np.clip(
            reg.predict(reg_scaler.transform(X_new[is_retained])), 0.0, None
        )
    return pred


n = len(X_int)
final_fracs = MLPR_predict_envelope(
    np.full(n, P),
    np.full(n, M_s),
    np.full(n, Z),
    np.full(n, M_c),
    np.full(n, rho),
    X_int,
)
print("MLPR predictions:", final_fracs)

# Clamp 0.0 (stripped) to 1e-4 so they appear on the log y-axis
final_fracs_plot = np.maximum(final_fracs, 1e-4)

plt.scatter(
    np.full(n, t_stop * 1e6),
    final_fracs_plot,
    marker="x",
    color="red",
    s=40,
    zorder=10,
    label="MLPR emulator",
)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Time [yrs]")
plt.ylabel("Envelope Mass Fraction")
plt.xlim(1e6, 8e9)   # Extended past 5e9 so markers aren't clipped
plt.ylim(8e-5, 1)    # Slightly below 1e-4 so floor markers are clearly visible
plt.title("Owen and Wu fig.2")
plt.legend()
plt.show()