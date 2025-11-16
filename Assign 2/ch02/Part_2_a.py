import numpy as np

# Load the data
X = []
y = []
with open('rectangle.data', 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        X.append([float(parts[0]), float(parts[1]), float(parts[2])])
        y.append(int(parts[3]))
X = np.array(X)
y = np.array(y)

# Perceptron parameters
w_init = np.array([0.25, -0.125, 0.0625])
b_init = 0.0
eta = 0.1


# For each epoch count
for n_epochs in [10, 20, 30, 100]:
    w = w_init.copy()
    b = b_init
    errors_per_epoch = []
    for epoch in range(n_epochs):
        errors = 0
        for xi, target in zip(X, y):
            pred = 1 if np.dot(w, xi) + b >= 0 else 0
            update = eta * (target - pred)
            if update != 0:
                w += update * xi
                b += update
                errors += 1
        errors_per_epoch.append(errors)
    print(f"Epochs: {n_epochs}, Final errors: {errors_per_epoch[-1]}")
    print(f"Final weights: {w}, Final bias: {b}")

    import matplotlib.pyplot as plt

# Error rates from part (a)
epochs = [10, 20, 30, 100]
errors = [0.65, 0.64, 0.65, 0.64]

plt.figure(figsize=(7,5))
plt.plot(epochs, errors, marker='o')
plt.title('Perceptron Error Rate vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show()

