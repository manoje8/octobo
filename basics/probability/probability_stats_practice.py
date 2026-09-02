"""
Probability & Statistics for ML — Hands-On Practice
=====================================================
Run each section to build intuition through visualization and simulation.

Usage:
    python probability_stats_practice.py

Requirements:
    pip install numpy matplotlib scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Consistent styling
plt.style.use("seaborn-v0_8-whitegrid")
np.random.seed(42)


# 1. PROBABILITY DISTRIBUTIONS — Visualize Them All


def plot_distributions():
    """Visualize the key distributions side by side."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Key Probability Distributions for ML", fontsize=16, fontweight="bold")

    # --- Bernoulli ---
    ax = axes[0, 0]
    p = 0.7
    ax.bar([0, 1], [1 - p, p], color=["#e74c3c", "#2ecc71"], width=0.4)
    ax.set_title(f"Bernoulli (p={p})")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Failure (0)", "Success (1)"])
    ax.set_ylabel("Probability")

    # --- Binomial ---
    ax = axes[0, 1]
    n, p = 20, 0.3
    x = np.arange(0, n + 1)
    ax.bar(x, stats.binom.pmf(x, n, p), color="#3498db", alpha=0.8)
    ax.axvline(n * p, color="red", linestyle="--", label=f"Mean = {n * p}")
    ax.set_title(f"Binomial (n={n}, p={p})")
    ax.set_xlabel("Number of successes")
    ax.legend()

    # --- Poisson ---
    ax = axes[0, 2]
    for lam in [2, 5, 10]:
        x = np.arange(0, 20)
        ax.plot(x, stats.poisson.pmf(x, lam), "o-", label=f"λ={lam}", markersize=4)
    ax.set_title("Poisson")
    ax.set_xlabel("Number of events")
    ax.legend()

    # --- Gaussian ---
    ax = axes[1, 0]
    x = np.linspace(-6, 6, 300)
    for mu, sigma in [(0, 1), (0, 2), (2, 1)]:
        ax.plot(x, stats.norm.pdf(x, mu, sigma), label=f"μ={mu}, σ={sigma}")
    ax.set_title("Gaussian (Normal)")
    ax.legend()

    # --- Exponential ---
    ax = axes[1, 1]
    x = np.linspace(0, 5, 300)
    for lam in [0.5, 1, 2]:
        ax.plot(x, stats.expon.pdf(x, scale=1 / lam), label=f"λ={lam}")
    ax.set_title("Exponential")
    ax.set_xlabel("Time between events")
    ax.legend()

    # --- Beta ---
    ax = axes[1, 2]
    x = np.linspace(0.01, 0.99, 300)
    for a, b in [(2, 5), (5, 2), (2, 2), (5, 5)]:
        ax.plot(x, stats.beta.pdf(x, a, b), label=f"α={a}, β={b}")
    ax.set_title("Beta (distribution over probabilities)")
    ax.set_xlabel("Probability p")
    ax.legend()

    plt.tight_layout()
    plt.savefig("distributions_overview.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: distributions_overview.png\n")


# 2. CENTRAL LIMIT THEOREM — Watch It Happen


def demonstrate_clt():
    """Show how the sum of ANY distribution converges to a Gaussian."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        "Central Limit Theorem: Sum of Random Variables → Gaussian",
        fontsize=14,
        fontweight="bold",
    )

    # Top row: Uniform distribution (very non-Gaussian!)
    for i, n_sum in enumerate([1, 2, 5, 30]):
        ax = axes[0, i]
        samples = np.mean(np.random.uniform(0, 1, (10000, n_sum)), axis=1)
        ax.hist(samples, bins=50, density=True, alpha=0.7, color="#3498db")
        ax.set_title(f"Mean of {n_sum} Uniform")
        if n_sum >= 5:
            x = np.linspace(samples.min(), samples.max(), 100)
            ax.plot(
                x,
                stats.norm.pdf(x, samples.mean(), samples.std()),
                "r-",
                linewidth=2,
                label="Gaussian fit",
            )
            ax.legend(fontsize=8)

    # Bottom row: Exponential distribution (skewed!)
    for i, n_sum in enumerate([1, 2, 5, 30]):
        ax = axes[1, i]
        samples = np.mean(np.random.exponential(1, (10000, n_sum)), axis=1)
        ax.hist(samples, bins=50, density=True, alpha=0.7, color="#e74c3c")
        ax.set_title(f"Mean of {n_sum} Exponential")
        if n_sum >= 5:
            x = np.linspace(samples.min(), samples.max(), 100)
            ax.plot(
                x,
                stats.norm.pdf(x, samples.mean(), samples.std()),
                "k-",
                linewidth=2,
                label="Gaussian fit",
            )
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("central_limit_theorem.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: central_limit_theorem.png")
    print("📝 Notice: Even very skewed distributions become Gaussian when averaged!\n")


# 3. BAYES' THEOREM — The Medical Test, Simulated


def simulate_bayes():
    """Simulate the medical test example with actual data."""
    print("=" * 60)
    print("BAYES' THEOREM: Medical Test Simulation")
    print("=" * 60)

    n_people = 1_000_000
    disease_rate = 0.001
    sensitivity = 0.99  # P(+|disease)
    false_positive = 0.05  # P(+|healthy)

    # Simulate the population
    has_disease = np.random.random(n_people) < disease_rate

    # Simulate test results
    test_positive = np.where(
        has_disease,
        np.random.random(n_people) < sensitivity,  # true positives
        np.random.random(n_people) < false_positive,  # false positives
    )

    # Compute P(disease | positive)
    positive_and_sick = np.sum(has_disease & test_positive)
    total_positive = np.sum(test_positive)

    simulated_prob = positive_and_sick / total_positive
    theoretical_prob = (sensitivity * disease_rate) / (
        sensitivity * disease_rate + false_positive * (1 - disease_rate)
    )

    print(f"\nPopulation size:        {n_people:,}")
    print(f"People with disease:    {np.sum(has_disease):,}")
    print(f"Total positive tests:   {total_positive:,}")
    print(f"True positives:         {positive_and_sick:,}")
    print(f"False positives:        {total_positive - positive_and_sick:,}")
    print(f"\nP(disease | positive):")
    print(f"  Simulated:   {simulated_prob:.4f} ({simulated_prob * 100:.2f}%)")
    print(f"  Theoretical: {theoretical_prob:.4f} ({theoretical_prob * 100:.2f}%)")
    print(f"\n💡 Even with a 99% accurate test, only ~{theoretical_prob * 100:.1f}% of")
    print(f"   positive results actually have the disease!\n")


# 4. BAYESIAN UPDATING — Watch Beliefs Change With Data


def bayesian_updating():
    """Flip a biased coin and watch the posterior distribution evolve."""
    true_p = 0.65  # True (unknown) bias
    flips = np.random.choice([0, 1], size=200, p=[1 - true_p, true_p])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Bayesian Updating: Estimating Coin Bias (true p = {true_p})",
        fontsize=14,
        fontweight="bold",
    )

    x = np.linspace(0, 1, 500)

    for idx, n_obs in enumerate([0, 1, 5, 20, 50, 200]):
        ax = axes[idx // 3, idx % 3]
        observed = flips[:n_obs]
        heads = int(np.sum(observed))
        tails = n_obs - heads

        # Prior: Beta(1,1) = Uniform
        # Posterior: Beta(1 + heads, 1 + tails)
        alpha_post = 1 + heads
        beta_post = 1 + tails
        posterior = stats.beta.pdf(x, alpha_post, beta_post)

        ax.plot(x, posterior, "b-", linewidth=2, label="Posterior")
        ax.axvline(
            true_p, color="red", linestyle="--", linewidth=2, label=f"True p={true_p}"
        )
        ax.fill_between(x, posterior, alpha=0.2)

        if n_obs > 0:
            mle = heads / n_obs
            ax.axvline(
                mle, color="green", linestyle=":", linewidth=2, label=f"MLE={mle:.2f}"
            )

        ax.set_title(f"After {n_obs} flips ({heads}H, {tails}T)")
        ax.set_xlabel("p (probability of heads)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("bayesian_updating.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: bayesian_updating.png")
    print(
        "📝 Notice: The posterior starts wide (uncertain) and narrows around the true value!\n"
    )


# 5. EXPECTATION & VARIANCE — The Bias-Variance Tradeoff Visualized


def bias_variance_demo():
    """Show bias-variance tradeoff with polynomial regression."""
    np.random.seed(42)

    # True function
    def true_function(x):
        return np.sin(2 * x) + 0.5 * x

    x_true = np.linspace(0, 4, 200)
    y_true = true_function(x_true)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Bias-Variance Tradeoff", fontsize=14, fontweight="bold")

    degrees = [1, 4, 15]
    titles = [
        "High Bias (Underfitting)\nDegree 1",
        "Good Balance\nDegree 4",
        "High Variance (Overfitting)\nDegree 15",
    ]

    for ax, deg, title in zip(axes, degrees, titles):
        ax.plot(x_true, y_true, "k-", linewidth=2, label="True function")

        # Fit model on multiple random datasets
        predictions = []
        for _ in range(20):
            x_train = np.random.uniform(0, 4, 20)
            y_train = true_function(x_train) + np.random.normal(0, 0.3, 20)

            coeffs = np.polyfit(x_train, y_train, deg)
            y_pred = np.polyval(coeffs, x_true)
            predictions.append(y_pred)
            ax.plot(x_true, y_pred, alpha=0.15, color="blue")

        # Mean prediction
        mean_pred = np.mean(predictions, axis=0)
        ax.plot(x_true, mean_pred, "r-", linewidth=2, label="Mean prediction")

        bias_sq = np.mean((mean_pred - y_true) ** 2)
        variance = np.mean(np.var(predictions, axis=0))

        ax.set_title(f"{title}\nBias²={bias_sq:.3f}, Var={variance:.3f}")
        ax.legend(fontsize=8)
        ax.set_ylim(-2, 5)

    plt.tight_layout()
    plt.savefig("bias_variance_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: bias_variance_tradeoff.png")
    print("📝 Blue lines = different training sets. Red = average prediction.")
    print("   Underfitting: predictions are consistent but wrong (high bias).")
    print("   Overfitting: predictions vary wildly (high variance).\n")


# 6. MLE IN ACTION — Fit a Gaussian, Then See Why MSE Works


def mle_gaussian():
    """Demonstrate MLE for Gaussian parameters."""
    print("=" * 60)
    print("MLE FOR GAUSSIAN: Finding μ and σ²")
    print("=" * 60)

    # Generate data from a known Gaussian
    true_mu, true_sigma = 5.0, 2.0
    data = np.random.normal(true_mu, true_sigma, 100)

    # MLE estimates
    mu_mle = np.mean(data)
    sigma_mle = np.std(data)  # Note: np.std uses N, not N-1 (biased)
    sigma_unbiased = np.std(data, ddof=1)

    print(f"\nTrue parameters:   μ = {true_mu}, σ = {true_sigma}")
    print(f"MLE estimates:     μ = {mu_mle:.4f}, σ = {sigma_mle:.4f}")
    print(f"Unbiased σ (ddof=1): σ = {sigma_unbiased:.4f}")
    print(f"\n💡 MLE of μ = sample mean (unbiased)")
    print(f"💡 MLE of σ² = sample variance with N divisor (slightly biased)")

    # Visualize: log-likelihood surface
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "MLE: Finding Parameters That Maximize Data Likelihood",
        fontsize=14,
        fontweight="bold",
    )

    # --- Log-likelihood as a function of μ (σ fixed) ---
    ax = axes[0]
    mu_range = np.linspace(3, 7, 200)
    log_likelihoods = [
        np.sum(stats.norm.logpdf(data, mu, true_sigma)) for mu in mu_range
    ]
    ax.plot(mu_range, log_likelihoods, "b-", linewidth=2)
    ax.axvline(
        mu_mle, color="red", linestyle="--", linewidth=2, label=f"MLE μ={mu_mle:.2f}"
    )
    ax.axvline(
        true_mu, color="green", linestyle=":", linewidth=2, label=f"True μ={true_mu}"
    )
    ax.set_xlabel("μ")
    ax.set_ylabel("Log-Likelihood")
    ax.set_title("Log-Likelihood vs μ (σ fixed)")
    ax.legend()

    # --- 2D log-likelihood surface ---
    ax = axes[1]
    mu_range = np.linspace(3.5, 6.5, 100)
    sigma_range = np.linspace(1.0, 3.0, 100)
    MU, SIGMA = np.meshgrid(mu_range, sigma_range)
    LL = np.zeros_like(MU)
    for i in range(len(sigma_range)):
        for j in range(len(mu_range)):
            LL[i, j] = np.sum(stats.norm.logpdf(data, MU[i, j], SIGMA[i, j]))

    contour = ax.contourf(MU, SIGMA, LL, levels=30, cmap="RdYlGn")
    ax.plot(
        mu_mle,
        sigma_mle,
        "r*",
        markersize=15,
        label=f"MLE ({mu_mle:.2f}, {sigma_mle:.2f})",
    )
    ax.plot(
        true_mu,
        true_sigma,
        "k*",
        markersize=15,
        label=f"True ({true_mu}, {true_sigma})",
    )
    ax.set_xlabel("μ")
    ax.set_ylabel("σ")
    ax.set_title("Log-Likelihood Surface")
    ax.legend()
    plt.colorbar(contour, ax=ax, label="Log-Likelihood")

    plt.tight_layout()
    plt.savefig("mle_gaussian.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n✅ Saved: mle_gaussian.png\n")


# 7. MLE → MSE CONNECTION: Why Squared Error Comes From Gaussian Assumption


def mle_equals_mse():
    """Show that MLE under Gaussian noise = minimizing MSE."""
    print("=" * 60)
    print("MLE = MSE: The Deep Connection")
    print("=" * 60)
    print("""
    Assume: y = f(x; θ) + ε,  where ε ~ N(0, σ²)

    Then:  P(y | x, θ) = N(y; f(x;θ), σ²)

    Log-likelihood:
        ℓ(θ) = Σ log P(yᵢ | xᵢ, θ)
             = Σ [ -½log(2πσ²) - (yᵢ - f(xᵢ;θ))² / (2σ²) ]

    Maximize ℓ(θ) ⟺ Minimize Σ (yᵢ - f(xᵢ;θ))²  ← This is MSE!

    The σ² and constants don't depend on θ, so they drop out.
    """)

    # Demonstrate with linear regression
    np.random.seed(42)
    n = 50
    x = np.random.uniform(0, 10, n)
    true_w, true_b = 2.5, 1.0
    noise_std = 2.0
    y = true_w * x + true_b + np.random.normal(0, noise_std, n)

    # Grid search over w and b
    w_range = np.linspace(1, 4, 100)
    b_range = np.linspace(-2, 4, 100)
    W, B = np.meshgrid(w_range, b_range)
    MSE = np.zeros_like(W)
    NLL = np.zeros_like(W)

    for i in range(len(b_range)):
        for j in range(len(w_range)):
            y_pred = W[i, j] * x + B[i, j]
            MSE[i, j] = np.mean((y - y_pred) ** 2)
            NLL[i, j] = -np.sum(stats.norm.logpdf(y, y_pred, noise_std))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "MLE under Gaussian Noise ≡ Minimizing MSE", fontsize=14, fontweight="bold"
    )

    for ax, surface, title in zip(
        axes, [MSE, NLL], ["MSE Surface", "Negative Log-Likelihood Surface"]
    ):
        contour = ax.contourf(W, B, surface, levels=30, cmap="RdYlGn_r")
        ax.plot(true_w, true_b, "k*", markersize=15, label=f"True ({true_w}, {true_b})")
        ax.set_xlabel("w (slope)")
        ax.set_ylabel("b (intercept)")
        ax.set_title(title)
        ax.legend()
        plt.colorbar(contour, ax=ax)

    plt.tight_layout()
    plt.savefig("mle_equals_mse.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: mle_equals_mse.png")
    print("📝 Both surfaces have their minimum at the same (w, b)!")
    print("   Minimizing MSE IS doing MLE under Gaussian noise.\n")


# 8. MLE vs MAP — Regularization as a Prior


def mle_vs_map():
    """Show how MAP with a Gaussian prior = MLE + L2 regularization."""
    print("=" * 60)
    print("MLE vs MAP: Regularization = Adding a Prior")
    print("=" * 60)

    np.random.seed(42)

    # Sparse data — overfitting is likely
    n = 8
    x = np.sort(np.random.uniform(0, 6, n))
    y = np.sin(x) + np.random.normal(0, 0.3, n)

    x_test = np.linspace(0, 6, 200)
    degree = 7  # High degree → overfitting

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("MLE vs MAP (Regularization)", fontsize=14, fontweight="bold")

    # Pure MLE (no regularization)
    ax = axes[0]
    coeffs = np.polyfit(x, y, degree)
    ax.scatter(x, y, color="red", zorder=5, label="Training data")
    ax.plot(x_test, np.polyval(coeffs, x_test), "b-", linewidth=2, label="MLE fit")
    ax.plot(x_test, np.sin(x_test), "g--", linewidth=2, label="True function")
    ax.set_title("MLE (No Regularization)\nOverfits!")
    ax.set_ylim(-3, 3)
    ax.legend(fontsize=8)
    max_coeff = np.max(np.abs(coeffs))
    ax.text(
        0.05,
        0.05,
        f"Max |coeff| = {max_coeff:.1f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    # MAP with light regularization (L2 / Ridge)
    for idx, alpha in enumerate([0.1, 10.0]):
        ax = axes[idx + 1]
        # Ridge regression: solve (X^T X + αI)w = X^T y
        X = np.vander(x, degree + 1)
        X_test = np.vander(x_test, degree + 1)
        I = np.eye(degree + 1)
        coeffs_map = np.linalg.solve(X.T @ X + alpha * I, X.T @ y)

        ax.scatter(x, y, color="red", zorder=5, label="Training data")
        ax.plot(
            x_test, X_test @ coeffs_map, "b-", linewidth=2, label=f"MAP (α={alpha})"
        )
        ax.plot(x_test, np.sin(x_test), "g--", linewidth=2, label="True function")
        ax.set_title(
            f"MAP / Ridge (α={alpha})\n{'Moderate' if alpha < 1 else 'Strong'} prior"
        )
        ax.set_ylim(-3, 3)
        ax.legend(fontsize=8)
        max_coeff = np.max(np.abs(coeffs_map))
        ax.text(
            0.05,
            0.05,
            f"Max |coeff| = {max_coeff:.2f}",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat"),
        )

    plt.tight_layout()
    plt.savefig("mle_vs_map.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved: mle_vs_map.png")
    print("📝 MLE overfits (large coefficients). MAP shrinks coefficients toward 0.")
    print("   The prior strength α controls how much shrinkage → regularization!\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Probability & Statistics for ML — Interactive Practice")
    print("=" * 60 + "\n")

    # Section 1: Visualize distributions
    print("📊 Section 1: Probability Distributions")
    plot_distributions()

    # Section 2: Central Limit Theorem
    print("📊 Section 2: Central Limit Theorem")
    demonstrate_clt()

    # Section 3: Bayes' Theorem simulation
    print("📊 Section 3: Bayes' Theorem")
    simulate_bayes()

    # Section 4: Bayesian updating
    print("📊 Section 4: Bayesian Updating")
    bayesian_updating()

    # Section 5: Bias-Variance Tradeoff
    print("📊 Section 5: Bias-Variance Tradeoff")
    bias_variance_demo()

    # Section 6: MLE for Gaussian
    print("📊 Section 6: MLE for Gaussian")
    mle_gaussian()

    # Section 7: MLE = MSE connection
    print("📊 Section 7: MLE = MSE Connection")
    mle_equals_mse()

    # Section 8: MLE vs MAP
    print("📊 Section 8: MLE vs MAP (Regularization)")
    mle_vs_map()

    print("\n" + "=" * 60)
    print(" All done! Check the generated PNG files for visualizations.")
    print("=" * 60)
