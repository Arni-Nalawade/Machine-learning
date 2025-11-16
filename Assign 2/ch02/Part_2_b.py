import numpy as np

# Load the data
data = []
with open('rectangle.data', 'r') as f:
    for line in f:
        items = [float(x) for x in line.strip().split(',')]
        data.append(items)
data = np.array(data)
X_train = data[:60, :3]
y_train = data[:60, 3]
X_test = data[60:, :3]
y_test = data[60:, 3]

# Perceptron training function
def train_perceptron(X, y, w_init, b_init, eta, n_epochs):
    w = w_init.copy()
    b = b_init
    for epoch in range(n_epochs):
        for xi, target in zip(X, y):
            pred = 1 if np.dot(w, xi) + b >= 0 else 0
            update = eta * (target - pred)
            w += update * xi
            b += update
    return w, b

# Initial values
w_init = np.array([0.25, -0.125, 0.0625])
b_init = 0.0
eta = 0.1

# Train
w, b = train_perceptron(X_train, y_train, w_init, b_init, eta, 10)

# Test in 8 groups of 5
group_results = []
for i in range(8):
    X_group = X_test[i*5:(i+1)*5]
    y_group = y_test[i*5:(i+1)*5]
    errors = 0
    for xi, target in zip(X_group, y_group):
        pred = 1 if np.dot(w, xi) + b >= 0 else 0
        if pred != target:
            errors += 1
    error_prop = errors / 5
    group_results.append(error_prop)

print("Error proportions per group:", group_results)
num_success_groups = sum(e <= 0.2 for e in group_results)
print("Groups with error ≤ 0.2:", num_success_groups)
print("Meets requirement (≥6 groups)?", num_success_groups >= 6)

import matplotlib.pyplot as plt
import numpy as np

# Error rates for each group in part (b)
group_errors = [0.6, 1.0, 0.6, 0.6, 0.4, 0.8, 0.4, 0.6]

plt.figure(figsize=(7,5))
plt.bar(np.arange(1,9), group_errors, color='lightpink')
plt.axhline(0.2, color='red', linestyle='--', label='ε = 0.2')
plt.title('Error Proportion per Test Group')
plt.xlabel('Test Group')
plt.ylabel('Error Proportion')
plt.legend()
plt.grid(True)
plt.show()

