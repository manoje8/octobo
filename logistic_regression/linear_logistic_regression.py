import numpy as np


class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.w = None
        self.b = 0.0


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.uniform(-5, 5, size=(200, 1))
    print(X.squeeze())
