"""
Calculus for ML — Hands-On Practice
====================================
Interactive exercises covering every concept from calculus-for-ml.md:
  1. Derivatives & numerical verification
  2. Partial derivatives for linear regression
  3. Chain rule through composed functions
  4. Gradients & gradient descent visualization
  5. Backpropagation from scratch (a 2-layer neural network)

Usage:
    python calculus_practice.py

Requirements:
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
np.random.seed(42)


# 1. DERIVATIVES — "If I Nudge the Input, What Happens to the Output?"


def section1_derivatives():
    """Visualize derivatives as slopes and verify them numerically."""
    print("=" * 65)
    print(" Section 1: DERIVATIVES — Sensitivity of Output to Input")
    print("=" * 65)

    # --- 1a: Numerical vs analytical derivative ---
    print("\n--- 1a: Numerical vs Analytical Derivative ---")
    print("For f(x) = x², the derivative is f'(x) = 2x.")
    print("Let's verify by nudging x and measuring the output change:\n")

    f = lambda x: x**2
    f_prime = lambda x: 2 * x

    h = 1e-7  # tiny nudge
    print(
        f"  {'x':>5}  {'f(x)':>10}  {'Analytical f′(x)':>18}  {'Numerical f′(x)':>18}  {'Match?':>8}"
    )
    print(f"  {'—' * 5}  {'—' * 10}  {'—' * 18}  {'—' * 18}  {'—' * 8}")
    for x in [1.0, 3.0, 5.0, -2.0]:
        analytical = f_prime(x)
        numerical = (f(x + h) - f(x)) / h
        match = "✅" if abs(analytical - numerical) < 1e-4 else "❌"
        print(
            f"  {x:5.1f}  {f(x):10.4f}  {analytical:18.6f}  {numerical:18.6f}  {match:>8}"
        )

    # --- 1b: Visualize common ML activation functions and their derivatives ---
    print("\n--- 1b: Activation Functions & Their Derivatives ---")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "ML Activation Functions & Their Derivatives", fontsize=15, fontweight="bold"
    )

    x = np.linspace(-5, 5, 500)

    # Sigmoid
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    sigmoid_deriv = lambda z: sigmoid(z) * (1 - sigmoid(z))

    ax = axes[0, 0]
    ax.plot(x, sigmoid(x), "b-", lw=2, label="σ(x)")
    ax.set_title("Sigmoid")
    ax.legend()
    ax.axhline(0, color="gray", lw=0.5)

    ax = axes[1, 0]
    ax.plot(x, sigmoid_deriv(x), "r-", lw=2, label="σ'(x) = σ(1-σ)")
    ax.axhline(0.25, color="gray", ls="--", lw=1, label="max = 0.25")
    ax.set_title("Sigmoid Derivative")
    ax.legend()
    ax.annotate(
        "← max gradient is only 0.25!\n   This causes vanishing gradients",
        xy=(0, 0.25),
        xytext=(1.5, 0.15),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9,
        color="red",
    )

    # Tanh
    tanh_deriv = lambda z: 1 - np.tanh(z) ** 2

    ax = axes[0, 1]
    ax.plot(x, np.tanh(x), "b-", lw=2, label="tanh(x)")
    ax.set_title("Tanh")
    ax.legend()
    ax.axhline(0, color="gray", lw=0.5)

    ax = axes[1, 1]
    ax.plot(x, tanh_deriv(x), "r-", lw=2, label="1 - tanh²(x)")
    ax.set_title("Tanh Derivative")
    ax.legend()

    # ReLU
    relu = lambda z: np.maximum(0, z)
    relu_deriv = lambda z: (z > 0).astype(float)

    ax = axes[0, 2]
    ax.plot(x, relu(x), "b-", lw=2, label="ReLU(x)")
    ax.set_title("ReLU")
    ax.legend()
    ax.axhline(0, color="gray", lw=0.5)

    ax = axes[1, 2]
    ax.plot(x, relu_deriv(x), "r-", lw=2, label="ReLU'(x)")
    ax.set_title("ReLU Derivative")
    ax.legend()
    ax.annotate(
        "Derivative is 0 or 1\n→ No vanishing gradient!",
        xy=(2, 1),
        xytext=(2, 0.5),
        arrowprops=dict(arrowstyle="->", color="green"),
        fontsize=9,
        color="green",
    )

    plt.tight_layout()
    plt.savefig("1_activation_derivatives.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: 1_activation_derivatives.png")
    print("📝 Key insight: Sigmoid's max derivative is 0.25 → gradients shrink")
    print("   exponentially through many layers (vanishing gradient problem).\n")


# 2. PARTIAL DERIVATIVES — "Which Knob Matters Most?"


def section2_partial_derivatives():
    """Compute and visualize partial derivatives for linear regression."""
    print("=" * 65)
    print(" Section 2: PARTIAL DERIVATIVES — One Knob at a Time")
    print("=" * 65)

    # --- 2a: Manual computation for L = (wx + b - y)² ---
    print("\n--- 2a: Partial Derivatives of Linear Regression Loss ---")
    print("Loss: L(w,b) = (wx + b - y)²")
    print("  ∂L/∂w = 2(wx + b - y) · x")
    print("  ∂L/∂b = 2(wx + b - y)")

    # Concrete values
    x_val, y_val = 3.0, 7.0
    w, b = 1.5, 0.5

    y_pred = w * x_val + b
    error = y_pred - y_val
    dL_dw_analytical = 2 * error * x_val
    dL_db_analytical = 2 * error

    # Numerical verification
    h = 1e-7
    L = lambda w_, b_: (w_ * x_val + b_ - y_val) ** 2
    dL_dw_numerical = (L(w + h, b) - L(w, b)) / h
    dL_db_numerical = (L(w, b + h) - L(w, b)) / h

    print(f"\n  x = {x_val}, y = {y_val}, w = {w}, b = {b}")
    print(f"  Prediction: ŷ = {w}×{x_val} + {b} = {y_pred}")
    print(f"  Error: ŷ - y = {error}")
    print(f"  Loss: L = {error}² = {error**2}")
    print(f"\n  {'':>20}  {'Analytical':>12}  {'Numerical':>12}  {'Match?':>8}")
    print(f"  {'—' * 20}  {'—' * 12}  {'—' * 12}  {'—' * 8}")
    match_w = "✅" if abs(dL_dw_analytical - dL_dw_numerical) < 1e-4 else "❌"
    match_b = "✅" if abs(dL_db_analytical - dL_db_numerical) < 1e-4 else "❌"
    print(
        f"  {'∂L/∂w':>20}  {dL_dw_analytical:12.6f}  {dL_dw_numerical:12.6f}  {match_w:>8}"
    )
    print(
        f"  {'∂L/∂b':>20}  {dL_db_analytical:12.6f}  {dL_db_numerical:12.6f}  {match_b:>8}"
    )

    # --- 2b: Visualize loss surface and gradient arrows ---
    print("\n--- 2b: Visualizing the Loss Landscape ---")

    # Generate a small dataset
    np.random.seed(42)
    X = np.random.uniform(0, 5, 20)
    Y = 2.0 * X + 1.0 + np.random.normal(0, 0.8, 20)  # true: w=2, b=1

    # Loss over a grid of (w, b)
    w_range = np.linspace(0, 4, 100)
    b_range = np.linspace(-2, 4, 100)
    W, B = np.meshgrid(w_range, b_range)
    Loss = np.zeros_like(W)
    for i in range(len(b_range)):
        for j in range(len(w_range)):
            predictions = W[i, j] * X + B[i, j]
            Loss[i, j] = np.mean((predictions - Y) ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Loss Landscape for Linear Regression", fontsize=14, fontweight="bold")

    # Contour plot
    ax = axes[0]
    contour = ax.contourf(W, B, Loss, levels=30, cmap="RdYlGn_r")
    plt.colorbar(contour, ax=ax, label="MSE Loss")
    ax.plot(2.0, 1.0, "k*", markersize=15, label="True (w=2, b=1)")
    ax.set_xlabel("w (weight)")
    ax.set_ylabel("b (bias)")
    ax.set_title("Loss Surface (contour)")
    ax.legend()

    # Gradient arrows at sample points
    for wi, bi in [(0.5, 0.5), (1.0, 3.0), (3.5, 0.0), (3.0, 3.0), (1.5, 1.5)]:
        preds = wi * X + bi
        grad_w = 2 * np.mean((preds - Y) * X)
        grad_b = 2 * np.mean((preds - Y))
        # Normalize for visualization
        norm = np.sqrt(grad_w**2 + grad_b**2)
        ax.arrow(
            wi,
            bi,
            -grad_w / norm * 0.3,
            -grad_b / norm * 0.3,
            head_width=0.08,
            head_length=0.05,
            fc="white",
            ec="white",
            lw=1.5,
        )

    # 3D surface
    ax = axes[1]
    ax = fig.add_subplot(122, projection="3d")
    ax.plot_surface(W, B, Loss, cmap="RdYlGn_r", alpha=0.8, edgecolor="none")
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_zlabel("Loss")
    ax.set_title("Loss Surface (3D)")
    ax.view_init(elev=30, azim=225)

    plt.tight_layout()
    plt.savefig("2_loss_surface.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: 2_loss_surface.png")
    print("📝 White arrows show −∇L (the direction gradient descent moves).")
    print("   They all point toward the minimum!\n")


# 3. CHAIN RULE — "How Do Effects Ripple Through a Pipeline?"


def section3_chain_rule():
    """Walk through the chain rule on composed functions."""
    print("=" * 65)
    print(" Section 3: CHAIN RULE — Tracking Effects Through Layers")
    print("=" * 65)

    # --- 3a: Simple chain rule example ---
    print("\n--- 3a: y = (3x + 2)² ---")
    print("  Inner: g(x) = 3x + 2")
    print("  Outer: y = g²")
    print("  Chain rule: dy/dx = 2g · 3 = 6(3x + 2)")

    f_composed = lambda x: (3 * x + 2) ** 2
    f_deriv = lambda x: 6 * (3 * x + 2)
    h = 1e-7

    print(
        f"\n  {'x':>5}  {'f(x)':>10}  {'Analytical':>12}  {'Numerical':>12}  {'Match?':>8}"
    )
    print(f"  {'—' * 5}  {'—' * 10}  {'—' * 12}  {'—' * 12}  {'—' * 8}")
    for x in [0.0, 1.0, 2.0, -1.0]:
        a = f_deriv(x)
        n = (f_composed(x + h) - f_composed(x)) / h
        match = "✅" if abs(a - n) < 1e-3 else "❌"
        print(f"  {x:5.1f}  {f_composed(x):10.4f}  {a:12.6f}  {n:12.6f}  {match:>8}")

    # --- 3b: Deeper chain — sigmoid(wx + b) ---
    print("\n--- 3b: Chain Rule Through a Neuron: a = σ(wx + b) ---")
    print("  z = wx + b   →  a = σ(z)   →  L = (a - y)²")
    print("  ∂L/∂w = ∂L/∂a · ∂a/∂z · ∂z/∂w")
    print("        = 2(a-y) · σ(z)(1-σ(z)) · x")

    sigmoid = lambda z: 1 / (1 + np.exp(-z))

    w, x_val, b, y_val = 0.5, 2.0, 0.1, 1.0
    z = w * x_val + b
    a = sigmoid(z)
    L_val = (a - y_val) ** 2

    dL_da = 2 * (a - y_val)
    da_dz = sigmoid(z) * (1 - sigmoid(z))
    dz_dw = x_val

    dL_dw_chain = dL_da * da_dz * dz_dw

    # Numerical check
    L_func = lambda w_: (sigmoid(w_ * x_val + b) - y_val) ** 2
    dL_dw_num = (L_func(w + h) - L_func(w)) / h

    print(f"\n  w={w}, x={x_val}, b={b}, y={y_val}")
    print(f"  z = wx+b = {z:.4f}")
    print(f"  a = σ(z) = {a:.4f}")
    print(f"  L = (a-y)² = {L_val:.4f}")
    print(f"\n  Chain rule breakdown:")
    print(f"    ∂L/∂a = 2(a-y) = {dL_da:.6f}")
    print(f"    ∂a/∂z = σ(z)(1-σ(z)) = {da_dz:.6f}")
    print(f"    ∂z/∂w = x = {dz_dw:.6f}")
    print(f"    ∂L/∂w = {dL_da:.6f} × {da_dz:.6f} × {dz_dw:.6f} = {dL_dw_chain:.6f}")
    print(f"    Numerical check: {dL_dw_num:.6f}")
    match = "✅" if abs(dL_dw_chain - dL_dw_num) < 1e-4 else "❌"
    print(f"    Match: {match}")

    # --- 3c: Visualize vanishing gradients through depth ---
    print("\n--- 3c: Vanishing Gradients Through Depth ---")
    print("  Each sigmoid layer multiplies gradient by at most 0.25.")
    print("  After n layers, gradient shrinks by 0.25^n:\n")

    depths = list(range(1, 16))
    sigmoid_decay = [0.25**d for d in depths]
    relu_decay = [1.0**d for d in depths]  # best case for ReLU

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(
        depths, sigmoid_decay, "ro-", lw=2, ms=8, label="Sigmoid (worst case: 0.25ⁿ)"
    )
    ax.semilogy(depths, relu_decay, "g^-", lw=2, ms=8, label="ReLU (best case: 1.0ⁿ)")
    ax.set_xlabel("Number of Layers", fontsize=12)
    ax.set_ylabel("Gradient Magnitude (log scale)", fontsize=12)
    ax.set_title(
        "Why Deep Sigmoid Networks Can't Learn: Vanishing Gradients",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.set_xticks(depths)

    for d in [5, 10, 15]:
        ax.annotate(
            f"{sigmoid_decay[d - 1]:.2e}",
            xy=(d, sigmoid_decay[d - 1]),
            xytext=(d + 0.5, sigmoid_decay[d - 1] * 5),
            fontsize=9,
            color="red",
            arrowprops=dict(arrowstyle="->", color="red"),
        )

    ax.axhline(1e-10, color="gray", ls="--", lw=1)
    ax.text(1, 2e-10, "Effectively zero — learning stops", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig("3_vanishing_gradients.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: 3_vanishing_gradients.png")
    print("📝 At 10 layers deep, sigmoid gradients are ~9.5e-7 → learning is")
    print(
        "   essentially frozen. This is why ReLU (and residual connections) matter.\n"
    )


# 4. GRADIENTS & GRADIENT DESCENT — "Walk Downhill"


def section4_gradient_descent():
    """Animate gradient descent on a 2D loss surface."""
    print("=" * 65)
    print(" Section 4: GRADIENT DESCENT — Walking Downhill")
    print("=" * 65)

    # --- 4a: Gradient descent on f(x,y) = x² + 3y² (elliptical bowl) ---
    print("\n--- 4a: Gradient Descent on f(x,y) = x² + 3y² ---")

    f = lambda x, y: x**2 + 3 * y**2
    grad_f = lambda x, y: np.array([2 * x, 6 * y])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Effect of Learning Rate on Gradient Descent", fontsize=14, fontweight="bold"
    )

    learning_rates = [0.01, 0.15, 0.35]
    lr_labels = ["Too small (lr=0.01)", "Just right (lr=0.15)", "Too large (lr=0.35)"]

    x_grid = np.linspace(-5, 5, 100)
    y_grid = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = f(X, Y)

    for ax, lr, label in zip(axes, learning_rates, lr_labels):
        ax.contour(X, Y, Z, levels=20, cmap="RdYlGn_r", alpha=0.6)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Run gradient descent
        pos = np.array([4.0, 4.0])  # starting point
        trajectory = [pos.copy()]

        for step in range(50):
            g = grad_f(pos[0], pos[1])
            pos = pos - lr * g
            trajectory.append(pos.copy())

            # Divergence check
            if np.any(np.abs(pos) > 100):
                break

        trajectory = np.array(trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], "b.-", ms=4, lw=1.5, label="Path")
        ax.plot(trajectory[0, 0], trajectory[0, 1], "rs", ms=10, label="Start")
        ax.plot(0, 0, "g*", ms=15, label="Minimum")

        if np.all(np.abs(trajectory[-1]) < 100):
            ax.plot(trajectory[-1, 0], trajectory[-1, 1], "k^", ms=10, label="End")

        ax.set_title(
            f"{label}\n{len(trajectory) - 1} steps, final loss = {f(trajectory[-1, 0], trajectory[-1, 1]):.4f}"
        )
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("4_learning_rate_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: 4_learning_rate_comparison.png")

    # --- 4b: Gradient descent for linear regression ---
    print("\n--- 4b: Gradient Descent Fitting a Line ---")

    np.random.seed(42)
    X_data = np.random.uniform(0, 5, 30)
    Y_data = 2.0 * X_data + 1.0 + np.random.normal(0, 1.0, 30)

    w, b = 0.0, 0.0
    lr = 0.02
    n = len(X_data)
    history = {"w": [w], "b": [b], "loss": []}

    for epoch in range(200):
        # Forward pass
        preds = w * X_data + b
        loss = np.mean((preds - Y_data) ** 2)
        history["loss"].append(loss)

        # Compute gradients (partial derivatives)
        dL_dw = (2 / n) * np.sum((preds - Y_data) * X_data)
        dL_db = (2 / n) * np.sum(preds - Y_data)

        # Update
        w -= lr * dL_dw
        b -= lr * dL_db
        history["w"].append(w)
        history["b"].append(b)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Gradient Descent: Linear Regression From Scratch",
        fontsize=14,
        fontweight="bold",
    )

    # Data + final fit
    ax = axes[0]
    ax.scatter(X_data, Y_data, c="steelblue", alpha=0.7, label="Data")
    x_line = np.linspace(0, 5, 100)
    ax.plot(x_line, 2.0 * x_line + 1.0, "g--", lw=2, label=f"True: y=2x+1")
    ax.plot(x_line, w * x_line + b, "r-", lw=2, label=f"Learned: y={w:.2f}x+{b:.2f}")
    ax.set_title("Fitted Line")
    ax.legend()

    # Loss curve
    ax = axes[1]
    ax.plot(history["loss"], "b-", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Loss Decreasing Over Training")

    # Parameter trajectory on loss surface
    ax = axes[2]
    w_range = np.linspace(-0.5, 3.5, 80)
    b_range = np.linspace(-1.5, 3.5, 80)
    WW, BB = np.meshgrid(w_range, b_range)
    LL = np.zeros_like(WW)
    for i in range(len(b_range)):
        for j in range(len(w_range)):
            LL[i, j] = np.mean((WW[i, j] * X_data + BB[i, j] - Y_data) ** 2)
    ax.contourf(WW, BB, LL, levels=25, cmap="RdYlGn_r", alpha=0.7)
    ws = history["w"]
    bs = history["b"]
    ax.plot(ws, bs, "w.-", ms=2, lw=1, alpha=0.9)
    ax.plot(ws[0], bs[0], "rs", ms=10, label="Start (0, 0)")
    ax.plot(ws[-1], bs[-1], "g*", ms=12, label=f"End ({ws[-1]:.2f}, {bs[-1]:.2f})")
    ax.set_xlabel("w")
    ax.set_ylabel("b")
    ax.set_title("Parameter Trajectory on Loss Surface")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("4_gradient_descent_linreg.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("✅ Saved: 4_gradient_descent_linreg.png")
    print(f"📝 After 200 epochs: w = {w:.4f} (true: 2.0), b = {b:.4f} (true: 1.0)")
    print(f"   Final loss: {history['loss'][-1]:.4f}\n")


# 5. BACKPROPAGATION — A Full 2-Layer Neural Network From Scratch


def section5_backprop():
    """Implement a 2-layer neural network with manual backprop."""
    print("=" * 65)
    print(" Section 5: BACKPROPAGATION — A Neural Network From Scratch")
    print("=" * 65)

    # --- 5a: Reproduce the exact walkthrough from the guide ---
    print("\n--- 5a: Exact Walkthrough From calculus-for-ml.md ---")
    print("  Network: x → [w₁, b₁] → σ → [w₂, b₂] → σ → L = (a₂ - y)²")
    print("  Values:  x=1, y=0, w₁=0.5, b₁=0.1, w₂=0.3, b₂=0.2\n")

    sigmoid = lambda z: 1 / (1 + np.exp(-z))

    x, y = 1.0, 0.0
    w1, b1 = 0.5, 0.1
    w2, b2 = 0.3, 0.2

    # Forward pass
    z1 = w1 * x + b1
    a1 = sigmoid(z1)
    z2 = w2 * a1 + b2
    a2 = sigmoid(z2)
    L = (a2 - y) ** 2

    print("  FORWARD PASS:")
    print(f"    z₁ = w₁·x + b₁ = {w1}×{x} + {b1} = {z1:.4f}")
    print(f"    a₁ = σ(z₁)     = σ({z1:.4f}) = {a1:.4f}")
    print(f"    z₂ = w₂·a₁+ b₂ = {w2}×{a1:.4f} + {b2} = {z2:.4f}")
    print(f"    a₂ = σ(z₂)     = σ({z2:.4f}) = {a2:.4f}")
    print(f"    L  = (a₂ - y)² = ({a2:.4f} - {y})² = {L:.4f}")

    # Backward pass
    dL_da2 = 2 * (a2 - y)
    da2_dz2 = sigmoid(z2) * (1 - sigmoid(z2))
    dz2_da1 = w2
    da1_dz1 = sigmoid(z1) * (1 - sigmoid(z1))
    dz1_dw1 = x

    dL_dw1 = dL_da2 * da2_dz2 * dz2_da1 * da1_dz1 * dz1_dw1

    print(f"\n  BACKWARD PASS:")
    print(f"    ∂L/∂a₂  = 2(a₂ - y)           = {dL_da2:.4f}")
    print(f"    ∂a₂/∂z₂ = σ(z₂)(1-σ(z₂))      = {da2_dz2:.4f}")
    print(f"    ∂z₂/∂a₁ = w₂                   = {dz2_da1:.4f}")
    print(f"    ∂a₁/∂z₁ = σ(z₁)(1-σ(z₁))      = {da1_dz1:.4f}")
    print(f"    ∂z₁/∂w₁ = x                    = {dz1_dw1:.4f}")
    print(
        f"\n    ∂L/∂w₁ = {dL_da2:.4f} × {da2_dz2:.4f} × {dz2_da1:.4f} × {da1_dz1:.4f} × {dz1_dw1:.4f}"
    )
    print(f"           = {dL_dw1:.4f}")

    # Verify with guide's value
    guide_value = 0.0197
    print(f"\n    Guide says: ∂L/∂w₁ ≈ 0.0197")
    print(f"    We got:     ∂L/∂w₁ = {dL_dw1:.4f}")
    match = "✅ Matches!" if abs(dL_dw1 - guide_value) < 0.001 else "❌ Mismatch"
    print(f"    {match}")

    # --- 5b: Full neural network training on XOR ---
    print("\n--- 5b: Training a Neural Network on XOR ---")
    print("  XOR is the classic problem that single-layer networks can't solve.")
    print("  We need at least one hidden layer.\n")

    # XOR dataset
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_xor = np.array([[0], [1], [1], [0]])

    # Network: 2 inputs → 4 hidden (sigmoid) → 1 output (sigmoid)
    np.random.seed(42)
    W1 = np.random.randn(2, 4) * 0.5
    B1 = np.zeros((1, 4))
    W2 = np.random.randn(4, 1) * 0.5
    B2 = np.zeros((1, 1))

    lr = 1.0
    losses = []

    for epoch in range(5000):
        # ---- FORWARD PASS ----
        Z1 = X_xor @ W1 + B1  # (4, 4)
        A1 = sigmoid(Z1)  # (4, 4)
        Z2 = A1 @ W2 + B2  # (4, 1)
        A2 = sigmoid(Z2)  # (4, 1)
        loss = np.mean((A2 - Y_xor) ** 2)
        losses.append(loss)

        # ---- BACKWARD PASS (chain rule at every step!) ----
        m = X_xor.shape[0]

        # ∂L/∂A2
        dL_dA2 = (2 / m) * (A2 - Y_xor)  # (4, 1)

        # ∂L/∂Z2 = ∂L/∂A2 · ∂A2/∂Z2   [chain rule]
        dA2_dZ2 = A2 * (1 - A2)  # sigmoid derivative
        dL_dZ2 = dL_dA2 * dA2_dZ2  # (4, 1)

        # ∂L/∂W2 = A1ᵀ · ∂L/∂Z2       [chain rule]
        dL_dW2 = A1.T @ dL_dZ2  # (4, 1)
        dL_dB2 = np.sum(dL_dZ2, axis=0, keepdims=True)  # (1, 1)

        # ∂L/∂A1 = ∂L/∂Z2 · W2ᵀ       [chain rule continues backward]
        dL_dA1 = dL_dZ2 @ W2.T  # (4, 4)

        # ∂L/∂Z1 = ∂L/∂A1 · ∂A1/∂Z1
        dA1_dZ1 = A1 * (1 - A1)
        dL_dZ1 = dL_dA1 * dA1_dZ1  # (4, 4)

        # ∂L/∂W1 = Xᵀ · ∂L/∂Z1
        dL_dW1 = X_xor.T @ dL_dZ1  # (2, 4)
        dL_dB1 = np.sum(dL_dZ1, axis=0, keepdims=True)  # (1, 4)

        # ---- UPDATE ----
        W2 -= lr * dL_dW2
        B2 -= lr * dL_dB2
        W1 -= lr * dL_dW1
        B1 -= lr * dL_dB1

    # Final predictions
    Z1 = X_xor @ W1 + B1
    A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + B2
    A2 = sigmoid(Z2)

    print("  Final predictions after 5000 epochs:")
    print(f"  {'Input':>10}  {'Target':>8}  {'Prediction':>12}  {'Rounded':>9}")
    print(f"  {'—' * 10}  {'—' * 8}  {'—' * 12}  {'—' * 9}")
    for i in range(4):
        pred = A2[i, 0]
        print(
            f"  {str(X_xor[i]):>10}  {Y_xor[i, 0]:8.0f}  {pred:12.4f}  {round(pred):9.0f}  {'✅' if round(pred) == Y_xor[i, 0] else '❌'}"
        )

    # --- Visualize training + decision boundary ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle(
        "Backpropagation: Training a Neural Network on XOR",
        fontsize=14,
        fontweight="bold",
    )

    # Loss curve
    ax = axes[0]
    ax.plot(losses, "b-", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss")
    ax.set_yscale("log")

    # Decision boundary
    ax = axes[1]
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z1_grid = grid @ W1 + B1
    A1_grid = sigmoid(Z1_grid)
    Z2_grid = A1_grid @ W2 + B2
    A2_grid = sigmoid(Z2_grid)
    ZZ = A2_grid.reshape(xx.shape)

    ax.contourf(xx, yy, ZZ, levels=50, cmap="RdYlBu", alpha=0.8)
    ax.contour(xx, yy, ZZ, levels=[0.5], colors="black", linewidths=2)
    colors = ["red" if y == 0 else "blue" for y in Y_xor.ravel()]
    ax.scatter(X_xor[:, 0], X_xor[:, 1], c=colors, s=200, edgecolors="black", zorder=5)
    ax.set_title("Decision Boundary")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")

    # Computation graph
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("The Backprop Pattern at Each Node", fontsize=12)

    # Draw a single node with incoming/outgoing gradients
    circle = plt.Circle(
        (5, 6), 1.2, fill=True, facecolor="#3498db", edgecolor="black", lw=2
    )
    ax.add_patch(circle)
    ax.text(
        5,
        6,
        "Node\nf(x)",
        ha="center",
        va="center",
        fontsize=11,
        color="white",
        fontweight="bold",
    )

    # Forward arrow
    ax.annotate(
        "",
        xy=(3.5, 6),
        xytext=(1, 6),
        arrowprops=dict(arrowstyle="-|>", color="green", lw=2.5),
    )
    ax.text(1, 6.5, "input x", fontsize=10, color="green")

    ax.annotate(
        "",
        xy=(9, 6),
        xytext=(6.5, 6),
        arrowprops=dict(arrowstyle="-|>", color="green", lw=2.5),
    )
    ax.text(7, 6.5, "output f(x)", fontsize=10, color="green")

    # Backward arrows
    ax.annotate(
        "",
        xy=(6.5, 4.5),
        xytext=(9, 4.5),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5),
    )
    ax.text(7, 3.8, "upstream\n  ∂L/∂f", fontsize=10, color="red")

    ax.annotate(
        "",
        xy=(1, 4.5),
        xytext=(3.5, 4.5),
        arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5),
    )
    ax.text(0.5, 3.8, "downstream\n∂L/∂f · ∂f/∂x", fontsize=10, color="red")

    ax.text(
        5,
        2,
        "Multiply upstream gradient\nby local derivative",
        ha="center",
        fontsize=11,
        fontweight="bold",
        style="italic",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange"
        ),
    )

    ax.text(
        5,
        9.2,
        "Green = Forward (left → right)\nRed = Backward (right → left)",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray"),
    )

    plt.tight_layout()
    plt.savefig("5_backprop_xor.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n✅ Saved: 5_backprop_xor.png")
    print("📝 XOR solved! The network learned a nonlinear decision boundary")
    print("   using the chain rule (backprop) to update all 4 layers of parameters.\n")


# =============================================================================
# 6. NUMERICAL GRADIENT CHECKING — Your Debugging Superpower
# =============================================================================


def section6_grad_check():
    """Teach numerical gradient checking as a debugging tool."""
    print("=" * 65)
    print(" Section 6: GRADIENT CHECKING — Verify Your Backprop!")
    print("=" * 65)
    print("\n  This is how practitioners debug backprop implementations.")
    print("  Compare your analytical gradient to a numerical approximation.\n")

    sigmoid = lambda z: 1 / (1 + np.exp(-z))

    # Build a small network
    np.random.seed(42)
    X = np.random.randn(3, 2)  # 3 samples, 2 features
    Y = np.array([[1], [0], [1]])

    W1 = np.random.randn(2, 3) * 0.5
    B1 = np.zeros((1, 3))
    W2 = np.random.randn(3, 1) * 0.5
    B2 = np.zeros((1, 1))

    def forward_and_loss(W1, B1, W2, B2):
        Z1 = X @ W1 + B1
        A1 = sigmoid(Z1)
        Z2 = A1 @ W2 + B2
        A2 = sigmoid(Z2)
        return np.mean((A2 - Y) ** 2)

    # Analytical gradients (backprop)
    Z1 = X @ W1 + B1
    A1 = sigmoid(Z1)
    Z2 = A1 @ W2 + B2
    A2 = sigmoid(Z2)
    m = X.shape[0]

    dL_dA2 = (2 / m) * (A2 - Y)
    dL_dZ2 = dL_dA2 * A2 * (1 - A2)
    dL_dW2 = A1.T @ dL_dZ2
    dL_dA1 = dL_dZ2 @ W2.T
    dL_dZ1 = dL_dA1 * A1 * (1 - A1)
    dL_dW1 = X.T @ dL_dZ1

    # Numerical gradients
    h = 1e-7
    num_dL_dW1 = np.zeros_like(W1)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_plus = W1.copy()
            W1_plus[i, j] += h
            W1_minus = W1.copy()
            W1_minus[i, j] -= h
            num_dL_dW1[i, j] = (
                forward_and_loss(W1_plus, B1, W2, B2)
                - forward_and_loss(W1_minus, B1, W2, B2)
            ) / (2 * h)

    # Compare
    print(
        f"  {'Parameter':>12}  {'Analytical':>12}  {'Numerical':>12}  {'Relative Diff':>14}"
    )
    print(f"  {'—' * 12}  {'—' * 12}  {'—' * 12}  {'—' * 14}")
    all_good = True
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            a = dL_dW1[i, j]
            n = num_dL_dW1[i, j]
            denom = max(abs(a) + abs(n), 1e-8)
            rel_diff = abs(a - n) / denom
            status = "✅" if rel_diff < 1e-5 else "❌"
            if rel_diff >= 1e-5:
                all_good = False
            print(
                f"  W1[{i},{j}]       {a:12.8f}  {n:12.8f}  {rel_diff:14.2e}  {status}"
            )

    print(
        f"\n  Overall: {'✅ All gradients match!' if all_good else '❌ Some gradients are off!'}"
    )
    print(f"\n  💡 Rule of thumb:")
    print(f"     Relative difference < 1e-5 → Great")
    print(f"     Relative difference < 1e-3 → Acceptable")
    print(f"     Relative difference > 1e-3 → Bug in your backprop!\n")


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  Calculus for ML — Interactive Practice")
    print("  (Companion to calculus-for-ml.md)")
    print("=" * 65 + "\n")

    section1_derivatives()
    section2_partial_derivatives()
    section3_chain_rule()
    section4_gradient_descent()
    section5_backprop()
    section6_grad_check()

    print("\n" + "=" * 65)
    print("  All done! Check the generated PNG files for visualizations:")
    print("    1_activation_derivatives.png  — Sigmoid vs ReLU derivatives")
    print("    2_loss_surface.png            — Loss landscape for linear regression")
    print("    3_vanishing_gradients.png     — Why deep sigmoid nets fail")
    print("    4_learning_rate_comparison.png— Effect of learning rate")
    print("    4_gradient_descent_linreg.png — GD fitting a line from scratch")
    print("    5_backprop_xor.png            — Full backprop on XOR")
    print("=" * 65 + "\n")
