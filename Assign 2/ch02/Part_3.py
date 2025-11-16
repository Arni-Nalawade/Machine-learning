import numpy as np

class AdalineGD:
    """
    ADAptive LInear NEuron classifier.
    Batch learning: weights updated after seeing ALL samples per epoch.
    Supports manual initialization of weights and bias.
    """
    def __init__(self, eta=0.01, n_iter=10, w_init=None, b_init=0.0):
        self.eta = eta
        self.n_iter = n_iter
        self.w_init = w_init
        self.b_init = b_init

    def fit(self, X, y):
        """Fit training data."""
        if self.w_init is not None:
            self.w_ = np.array(self.w_init, dtype=np.float64)
        else:
            self.w_ = np.zeros(X.shape[1], dtype=np.float64)
        self.b_ = self.b_init
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = net_input  # linear activation
            errors = y - output
            # Batch gradient update
            self.w_ += self.eta * (X.T @ errors) / X.shape[0]
            self.b_ += self.eta * errors.mean()
            loss = (errors ** 2).mean()
            self.losses_.append(loss)
        return self
    
    def net_input(self, X):
        """Calculate net input."""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Predict binary class label using threshold 0.5."""
        return np.where(self.net_input(X) >= 0.5, 1, 0)


if __name__ == '__main__':
    # Inputs and targets (truth tables) for each function
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    targets = {
        'AND':      np.array([0, 0, 0, 1]),
        'OR':       np.array([0, 1, 1, 1]),
        'IMPLIES':  np.array([1, 1, 0, 1])
    }

    for func_name, y in targets.items():
        for eta in [0.01, 0.05, 0.1, 0.5, 0.75]:
            model = AdalineGD(eta=eta, n_iter=10, w_init=[0.25, -0.125], b_init=0.0)
            model.fit(X, y)
            predictions = model.predict(X)
            all_correct = np.array_equal(predictions, y)
            print(f"Function: {func_name}, eta: {eta}")
            print(f"MSE per epoch: {model.losses_}")
            print(f"Predictions: {predictions}")
            print(f"All correct? {all_correct}")
            print(f"Final weights: {model.w_}, bias: {model.b_}")
            print('-' * 40)

import numpy as np
import matplotlib.pyplot as plt

# Boolean function data
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
targets_dict = {
    'AND': np.array([0,0,0,1]),
    'OR': np.array([0,1,1,1]),
    'IMPLIES': np.array([1,1,0,1])
}

class AdalineGD:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y, w_init, b_init):
        self.w_ = np.array(w_init)
        self.b_ = b_init
        self.cost_ = []
        for _ in range(self.n_iter):
            net_input = np.dot(X, self.w_) + self.b_
            errors = y - net_input
            self.w_ += self.eta * X.T.dot(errors)
            self.b_ += self.eta * errors.sum()
            cost = (errors**2).mean()
            self.cost_.append(cost)
        return self

learning_rates = [0.01, 0.05, 0.1, 0.5, 0.75]
w_init = [0.25, -0.125]
b_init = 0.0
n_iter = 10

for func_name, targets in targets_dict.items():
    plt.figure(figsize=(7,5))
    for eta in learning_rates:
        ada = AdalineGD(eta=eta, n_iter=n_iter)
        ada.fit(inputs, targets, w_init, b_init)
        plt.plot(range(1, n_iter+1), ada.cost_, marker='o', label=f'η={eta}')
    plt.title(f'AdalineGD MSE for {func_name} Function')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.show()
