import numpy as np
#import os
#print('Current working directory:', os.getcwd())

def train_perceptron_boolean(X, y, w_init, b_init, eta=0.1, max_epochs=10):
    """
    Train a Perceptron on 2-input boolean data.

    Parameters:
    - X: np.array of shape (4, 2), input vectors
    - y: np.array of shape (4,), target labels (0 or 1)
    - w_init: initial weights, np.array of shape (2,)
    - b_init: initial bias, scalar
    - eta: learning rate
    - max_epochs: maximum number of epochs to train

    Returns:
    - converged: bool, whether training converged within max_epochs
    - epochs: int, number of epochs taken to converge (or max_epochs if not)
    - w: final weights np.array
    - b: final bias scalar
    - correct_boundary: bool, whether final weights/bias classify all inputs correctly
    """
    w = w_init.copy()
    b = b_init
    n_samples = X.shape[0]

    def predict(x):
        return 1 if np.dot(w, x) + b >= 0 else 0

    for epoch in range(1, max_epochs + 1):
        errors = 0
        for xi, target in zip(X, y):
            pred = predict(xi)
            update = eta * (target - pred)
            if update != 0:
                w += update * xi
                b += update
                errors += 1
        if errors == 0:
            # Converged
            break

    # Check correctness of final boundary
    correct_boundary = all(predict(xi) == target for xi, target in zip(X, y))

    converged = (errors == 0)
    return converged, epoch, w, b, correct_boundary

if __name__ == '__main__':
    # Define inputs: all 4 combinations of 2 boolean variables
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    # Define initial weights and bias as per instructions
    w_init = np.array([0.25, -0.125])
    b_init = 0.0
    eta = 0.1
    max_epochs = 10

    # Define target outputs for each boolean function
    targets = {
        'AND': np.array([0, 0, 0, 1]),
        'OR': np.array([0, 1, 1, 1]),
        'IMPLIES': np.array([1, 1, 0, 1])  # x1 implies x2
    }

    # Train and report for each function
    for func_name, y in targets.items():
        converged, epochs, w_final, b_final, correct_boundary = train_perceptron_boolean(
            X, y, w_init, b_init, eta, max_epochs
        )
        print(f"Function: {func_name}")
        print(f"Converged: {converged}")
        print(f"Epochs to converge: {epochs}")
        print(f"Final weights: {w_final}")
        print(f"Final bias: {b_final}")
        print(f"Correct boundary: {correct_boundary}")
        print("-" * 30)
