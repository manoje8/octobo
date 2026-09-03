import numpy as np

np.random.seed(42)


class TinyNN:
    def __init__(self, layer_sizes):
        """
        layer_sizes: e.g. [2, 4, 4, 1] -> input=2, hidden=4, hidden=4, output=1
        len(layer_sizes) - 1 = number of weight matrices ("layers" of computation).
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(
                2.0 / layer_sizes[i]
            )
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_deriv(x):
        return (x > 0).astype(float)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, X):
        """
        Computes the output AND caches every z (pre-activation) and a
        (post-activation) along the way — backward() needs these for the
        chain rule.
        """
        self.z = []
        self.a = [X]

        for i in range(self.num_layers):
            z = self.a[-1] @ self.weights[i] + self.biases[i]
            self.z.append(z)
            is_output_layer = i == self.num_layers - 1
            a = self.sigmoid(z) if is_output_layer else self.relu(z)
            self.a.append(a)

        return self.a[-1]

    def backward(self, y, learning_rate):
        """
        Manual backpropagation. Walks the layers in reverse, applying the
        chain rule one layer at a time:

            dL/dz[last]  = y_pred - y_true      (sigmoid + BCE loss combine nicely)
            dL/dW[i]     = a[i].T @ dz[i] / m
            dL/db[i]     = mean(dz[i])
            dL/da[i]     = dz[i] @ W[i].T
            dL/dz[i-1]   = dL/da[i] * relu'(z[i-1])   (only for hidden layers)
        """

        m = y.shape[0]
        y_pred = self.a[-1]

        # gradient at the output layer (BCE + sigmoid shortcut)
        dz = y_pred - y

        grads_w, grads_b = [None] * self.num_layers, [None] * self.num_layers

        for i in reversed(range(self.num_layers)):
            a_prev = self.a[i]
            grads_w[i] = a_prev.T @ dz / m
            grads_b[i] = np.sum(dz, axis=0, keepdims=True) / m

            if i > 0:
                da_prev = dz @ self.weights[i].T
                dz = da_prev * self.relu_deriv(self.z[i - 1])

        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * grads_w[i]
            self.biases[i] -= learning_rate * grads_b[i]

        return grads_w, grads_b

    @staticmethod
    def bce_loss(y_true, y_pred, eps=1e-9):
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def train(self, X, y, epochs=5000, learning_rate=0.5, verbose=True, log_every=500):
        losses = []

        for i in range(epochs):
            y_pred = self.forward(X)
            loss = self.bce_loss(y, y_pred)
            losses.append(loss)
            self.backward(y, learning_rate)

            if verbose and i % log_every == 0:
                print(f"epoch {epochs:5d} | loss {loss:.4f}")

        return losses


def numerical_gradient_check(net, x, y, epsilon=1e-5):
    """
    For a handful of weights, wiggle them by +-epsilon, rerun the forward
    pass, and estimate dL/dw as a finite difference. Compare that estimate
    to what backward() computed analytically. If they match to ~1e-6,
    the backprop implementation is correct.
    """

    net.forward(x)

    # lr=0 so weights don't actually move
    analytic_w, _ = net.backward(y, learning_rate=0.0)

    print("\nGradient check (analytic vs numerical, should match closely):")

    rng = np.random.default_rng(0)

    for layer_idx in range(net.num_layers):
        W = net.weights[layer_idx]
        flat_idx_options = rng.choice(W.size, size=min(3, W.size), replace=False)

        for flat_idx in flat_idx_options:
            idx = np.unravel_index(flat_idx, W.shape)
            original_value = W[idx]

            W[idx] = original_value + epsilon
            loss_plus = net.bce_loss(y, net.forward(x))

            W[idx] = original_value - epsilon
            loss_minus = net.bce_loss(y, net.forward(x))

            W[idx] = original_value

            numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
            analytic_grad = analytic_w[layer_idx][idx]

            diff = abs(numerical_grad - analytic_grad)

            print(
                f" layer {layer_idx} weight{idx}: analytic={analytic_grad:.6f} "
                f"numerical={numerical_grad: .6f} diff={diff:.2e}"
            )


if __name__ == "__main__":
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([[0], [1], [1], [0]], dtype=float)

    net = TinyNN([2, 4, 4, 1])

    print("Training on XOR...")
    losses = net.train(x, y, epochs=5000, learning_rate=0.5)

    print("\nFinal prediction...")
    preds = net.forward(x)

    for xi, yi, pi in zip(x, y, preds):
        print(f" input={xi} --> output={yi[0]:.0f}  pred={pi[0]:.4f}")

    numerical_gradient_check(net, x, y)

    np.savetxt("losses.csv", losses, delimiter=",", fmt="%10.6f")
