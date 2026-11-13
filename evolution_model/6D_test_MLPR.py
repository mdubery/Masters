from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Load the dataset 
data = pd.read_csv('/home/matthew/Documents/Masters/code/Masters/evolution_model/training_data_6D.csv')
Period = data["Period (days)"]
Mass = data["Stellar Mass (M_sun)"]
Metalicity = data["Metalicity"]
M_core = data["M_core"] 
rho = data["rho"]
X_int = data["Log x int"]

#Y = data['Final Fraction'].to_numpy()

X = np.column_stack((np.log10(Period), Mass, Metalicity, M_core, rho, np.log10(X_int)))


Y_frac = data["Final Envelope Fraction"].to_numpy()
Y = (data["Final Envelope Fraction"] > 1e-3).astype(int)  
#Final Radius (R_earth)

# 2. Split into 80% training data and 20% test data
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# 3. Scale 
x_scaler = StandardScaler()
X_train_scaled = x_scaler.fit_transform(X_train)
X_test_scaled = x_scaler.transform(X_test)
# 4. Instantiate and train the Multi-Layer Perceptron
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32), 
    activation='relu',           
    max_iter=3000,                
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# 5. Evaluate the model performance
predictions = mlp.predict(X_test_scaled)
print(f"Overall Accuracy: {accuracy_score(y_test, predictions):.2%}\n")
print(classification_report(y_test, predictions))

acc = accuracy_score(y_test, predictions)


# 6. Graph for classifier
p_grid = np.geomspace(1.0, 100.0, 200)
m_grid = np.linspace(0.5, 2.0, 200)
PP, MM = np.meshgrid(p_grid, m_grid)

n_pts = PP.size
grid_6d = np.column_stack((
    np.log10(PP.ravel()),
    MM.ravel(),
    np.full(n_pts, 0.0),            # fixed Metalicity
    np.full(n_pts, 10.0),           # fixed M_core
    np.full(n_pts, 0.33),           # fixed rho
    np.full(n_pts, np.log10(0.03)), # fixed initial X
))
grid_scaled = x_scaler.transform(grid_6d)
probs = mlp.predict_proba(grid_scaled)[:, 1].reshape(PP.shape)

plt.figure(figsize=(7, 5))
plt.contourf(PP, MM, probs, levels=20, cmap="RdBu", alpha=0.8)
plt.colorbar(label="P(Retained Envelope)")
plt.xscale("log")
plt.xlabel("Orbital Period (days)")
plt.ylabel("Stellar Mass ($M_\\odot$)")
plt.title("Photoevaporation Boundary (100-Run Prototype)")
plt.tight_layout()
plt.show()

#1. Load
#for classified as retained envelope (1) and lost envelope (0)
retained_indices = np.where(Y == 1)[0]
lost_indices = np.where(Y == 0)[0]

#new training data with only retained
A = X[retained_indices]
B = Y_frac[retained_indices]

#2. train test split
A_train, A_test, B_train, B_test = train_test_split(
    A, B, test_size=0.2, random_state=42
)

#3. scale
A_scaler = StandardScaler()
A_train_scaled = A_scaler.fit_transform(A_train)
A_test_scaled = A_scaler.transform(A_test)


#4. initate and run
mlpr = MLPRegressor(
    hidden_layer_sizes=(100, 100, 50, 15), 
    activation='relu', 
    solver='adam', 
    learning_rate_init=0.001, 
    max_iter=3000, 
    random_state=42,
    tol = 1e-8,
    alpha = 1e-6
    )
mlpr.fit(A_train_scaled, B_train)

#5. validate
predictions_r = mlpr.predict(A_test_scaled)
print(f"R² Score: {r2_score(B_test, predictions_r):.4f}")
print(f"Mean Absolute Error: {mean_absolute_error(B_test, predictions_r):.6f}")

#6. MLPR plots

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

#Predicted vs. True
preds_plot = np.clip(predictions_r, 1e-4, None)
ax1.scatter(B_test, preds_plot, color="tab:blue", alpha=0.3, s=10)
low, high = 1e-3, max(B_test.max(), preds_plot.max()) * 1.1
ax1.plot([low, high], [low, high], "k--", label="Ideal 1:1 Fit")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlim(low, high)
ax1.set_ylim(low, high)
ax1.set_xlabel("True Final Envelope Fraction (Simulation)")
ax1.set_ylabel("Predicted Final Envelope Fraction (mlpr)")
ax1.set_title(f"Stage 2 Test Parity (R² = {r2_score(B_test, predictions_r):.4f})")
ax1.legend()

# Panel 2: Full two-stage emulator map (mlp mask + mlpr values) vs. simulations
is_retained = mlp.predict(x_scaler.transform(grid_6d)) == 1

pred_frac_grid = np.zeros(len(grid_6d))
if np.any(is_retained):
    pred_frac_grid[is_retained] = np.clip(
        mlpr.predict(A_scaler.transform(grid_6d[is_retained])), 0.0, None
    )


pred_frac_grid = pred_frac_grid.reshape(PP.shape)

cf = ax2.contourf(PP, MM, pred_frac_grid, levels=20, cmap="viridis")
plt.colorbar(cf, ax=ax2, label="Predicted Final Envelope Fraction")

ax2.set_xscale("log")
ax2.set_xlabel("Orbital Period (days)")
ax2.set_ylabel("Stellar Mass ($M_\\odot$)")
ax2.set_title("Two-Stage Emulator vs. Simulation Data")

plt.tight_layout()
plt.show()