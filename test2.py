from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt
import numpy as np

# 1. Load the dataset (8x8 pixel images of handwritten digits)
digits = load_digits()
X, y = digits.data, digits.target

# 2. Split into 80% training data and 20% test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale feature pixel values (Crucial for Neural Networks!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Instantiate and train the Multi-Layer Perceptron
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32), # 2 hidden layers with 64 and 32 neurons
    activation='relu',           # Standard activation function
    max_iter=300,                # Maximum iterations/epochs
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# 5. Evaluate the model performance
predictions = mlp.predict(X_test_scaled)
#print(f"Overall Accuracy: {accuracy_score(y_test, predictions):.2%}\n")
#print(classification_report(y_test, predictions))

acc = accuracy_score(y_test, predictions)


def run(a, b):
    mlp = MLPClassifier(
        hidden_layer_sizes=(a, b),  # Use parameters a and b for hidden layers
        activation='relu',
        max_iter=300,
        random_state=42
    )
    mlp.fit(X_train_scaled, y_train)
    predictions = mlp.predict(X_test_scaled)
    acc = accuracy_score(y_test, predictions)
    return acc

s = 2

a = np.arange(10, 100, s)
b = np.arange(10, 100, s)

x, y = np.meshgrid(a, b)
z = np.zeros((len(a), len(b)))

for i in range (len(a)):
    for j in range(len(b)):
        acc = run(a[i], b[j])
        #percent done 
        print(f"Percent done: {((i*len(b) + j + 1)/(len(a)*len(b)))*100:.2f}%")

        z[j, i] = acc

plt.style.use('_mpl-gallery-nogrid')
fig, ax = plt.subplots()

levels = np.linspace(np.min(z), np.max(z), 20)
contour = ax.contourf(x, y, z, levels=levels)

cbar = fig.colorbar(contour, ax=ax)
cbar.set_label('Accuracy Score')
ax.set_xlabel('Layer 1 Neurons (a)')
ax.set_ylabel('Layer 2 Neurons (b)')
ax.set_title('MLP Hyperparameter Grid Search Accuracy')


plt.show()
