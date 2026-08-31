"""
Principal Component Analysis (PCA) from scratch, using only NumPy and SVD.

Why SVD instead of eigendecomposing the covariance matrix?
------------------------------------------------------------
The "textbook" approach computes the covariance matrix C = Xc.T @ Xc / (n-1)
and eigendecomposes it. That works, but squaring the data to form C amplifies
numerical error (condition number gets squared) and costs more when
n_features is large. Computing the SVD of the centered data directly avoids
ever forming C and is what libraries like scikit-learn do internally.

Math recap
----------
Given data X of shape (n_samples, n_features):

1. Center the data:   Xc = X - mean(X, axis=0)

2. SVD: Xc = U @ diag(S) @ Vt

   - Rows of Vt (== columns of V) are the principal directions. They are
     exactly the eigenvectors of the covariance matrix Xc.T @ Xc.
   - Singular values S relate to the eigenvalues (variances) of the
     covariance matrix by:
         explained_variance_i = S_i^2 / (n_samples - 1)

3. Projecting onto the top k components:
         X_pca = Xc @ Vt[:k].T   ==   U[:, :k] @ diag(S[:k])

4. Reconstructing from k components:
         X_approx = X_pca @ Vt[:k] + mean
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PCA:
    """
    Principal Component Analysis implemented via SVD.

     Parameters
     ----------
     n_components : int, optional
         Number of components to keep. If None, keep all
         min(n_samples, n_features) components.

     Attributes (set after calling fit)
     -----------------------------------
     mean_ : ndarray of shape (n_features,)
         Per-feature mean of the training data (used to center new data).
     components_ : ndarray of shape (n_components, n_features)
         Principal axes, sorted by decreasing explained variance. Each row
         is a unit vector.
     singular_values_ : ndarray of shape (n_components,)
         Singular values corresponding to each component.
     explained_variance_ : ndarray of shape (n_components,)
         Variance explained by each component.
     explained_variance_ratio_ : ndarray of shape (n_components,)
         Fraction of total variance explained by each component.
    """

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.mean = None
        self.components_ = None
        self.singular_values_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        print(f"Fitting PCA with {n_samples} samples and {n_features} features")

        max_components = min(n_samples, n_features)
        print(f"Maximum number of components is {max_components}")

        n_components = self.n_components or max_components
        print(f"Using {n_components} components for PCA")

        if n_components > max_components:
            raise ValueError(
                f"n_components={n_components} cannot exceed "
                f"min(n_samples, n_features)={max_components}"
            )

        # center the data
        self.mean = np.mean(X, axis=0)
        Xc = X - self.mean

        # SVD of centered data
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        # Sign convention
        max_abs_cols = np.argmax(np.abs(Vt), axis=1)
        signs = np.sign(Vt[np.arange(Vt.shape[0]), max_abs_cols])
        signs[signs == 0] = 1.0

        Vt = Vt * signs[:, np.newaxis]
        U = U * signs[np.newaxis, :]

        self.components_ = Vt[:n_components]
        self.singular_values_ = S[:n_components]

        explained_variance = (S**2) / (n_samples - 1)
        total_variance = explained_variance.sum()
        self.explained_variance_ = explained_variance[:n_components]
        self.explained_variance_ratio_ = (
            explained_variance[:n_components] / total_variance
        )

        self._n_samples_fit = n_samples
        return self

    def transform(self, X):
        if self.components_ is None:
            raise RuntimeError("PCA instance is not fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=float)
        Xc = X - self.mean
        return Xc @ self.components_.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_pca):
        """Approximately reconstruct original-space data from projections."""

        if self.components_ is None:
            raise RuntimeError("PCA instance is not fitted yet. Call fit() first.")

        X_pca = np.asarray(X_pca, dtype=float)
        return X_pca @ self.components_ + self.mean

    def plot_pca_components(self, X, pca, n_components: int = 2):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c="orange", alpha=0.5, s=10)
        ax.scatter(
            self.mean[0],
            self.mean[1],
            self.mean[2],
            c="red",
            s=100,
            marker="X",
            label="Mean",
        )

        colors = ["red", "blue", "green", "orange", "purple"]
        scale = 3

        for i in range(min(n_components, self.components_.shape[0])):
            comp = self.components_[i]
            n = np.sqrt(self.explained_variance_[i]) * 2
            ax.quiver(
                self.mean[0],
                self.mean[1],
                self.mean[2],
                comp[0] * n,
                comp[1] * n,
                comp[2] * n,
                color=colors[i],
                arrow_length_ratio=0.1,
                label=f"PC{i + 1} (var{self.explained_variance_ratio_[i]:2f})",
            )

        ax.set_title("Principal component in original space")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")

        ax.legend()
        plt.show()

    def plot_scree(self):
        """Plot explained variance for each component"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Individual explained variance
        components = np.arange(1, len(self.explained_variance_ratio_) + 1)
        ax1.bar(components, self.explained_variance_ratio_)
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Explained Variance Ratio")
        ax1.set_title("Individual Explained Variance")
        ax1.grid(True, alpha=0.3)

        # Cumulative explained variance
        cumsum = np.cumsum(self.explained_variance_ratio_)
        ax2.plot(components, cumsum, "bo-", linewidth=2)
        ax2.axhline(y=0.95, color="r", linestyle="--", label="95% threshold")
        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Explained Variance")
        ax2.set_title("Cumulative Explained Variance")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_reconstruction_error(self, X, max_components=None):
        """Plot reconstruction error as a function of number of components"""
        if max_components is None:
            max_components = min(X.shape)

        errors = []
        for k in range(1, max_components + 1):
            # Create new PCA with k components
            pca_k = PCA(n_components=k)
            X_pca_k = pca_k.fit_transform(X)
            X_recon = pca_k.inverse_transform(X_pca_k)
            mse = np.mean((X - X_recon) ** 2)
            errors.append(mse)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_components + 1), errors, "bo-", linewidth=2)
        plt.xlabel("Number of Components")
        plt.ylabel("Reconstruction MSE")
        plt.title("Reconstruction Error vs Number of Components")
        plt.grid(True, alpha=0.3)

        # Mark the current n_components
        if hasattr(self, "components_"):
            k = self.components_.shape[0]
            plt.axvline(
                x=k, color="r", linestyle="--", label=f"Current: {k} components"
            )
            plt.legend()

        plt.show()

    def plot_component_heatmap(self):

        plt.figure(figsize=(10, 6))
        plt.imshow(self.components_, cmap="RdBu_r", aspect="auto")
        plt.colorbar(label="Component weight")
        plt.xlabel("Features")
        plt.ylabel("Principal Components")
        plt.title("Principal Components Heatmap")

        for i in range(self.components_.shape[0]):
            for j in range(self.components_.shape[1]):
                plt.text(
                    j,
                    i,
                    f"{self.components_[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        plt.show()


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n_samples = 500

    z1 = rng.normal(0, 3, n_samples)
    z2 = rng.normal(0, 1, n_samples)
    noise = rng.normal(0, 0.5, (n_samples, 3))

    basic = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

    X = np.outer(z1, basic[0]) + np.outer(z2, basic[1]) + noise

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    print("Explained variance ratio: ", pca.explained_variance_ratio_)
    print("Sum of first 2 components' ratio: ", pca.explained_variance_ratio_.sum())
    print("Component shape: ", pca.components_.shape)
    print("Projected data shape: ", X_pca.shape)

    X_reconstructed = pca.inverse_transform(X_pca)
    reconstruction_mse = np.mean((X - X_reconstructed) ** 2)

    print(f"Reconstruction MSE: {reconstruction_mse:.6f}")

    # Cross-check against the covariance-matrix eigendecomposition
    Xc = X - X.mean(axis=0)
    cov = (Xc.T @ Xc) / (n_samples - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    print("\nExplained variance (SVD): ", pca.explained_variance_)
    print("Top eigenvalues (eig on cov): ", eigvals[:2])

    # Compare subspaces via absolute cosine similarity (sign-invariant)
    for i in range(2):
        cos_sim = np.abs(np.dot(pca.components_[i], eigvecs[:, i]))
        print(
            f"component {i} alignment in eigen composition: {cos_sim:.6f}"
            f"(1.0 = identical direction)"
        )

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection="3d")
    scatter1 = ax1.scatter(
        X[:, 0], X[:, 1], X[:, 2], c=X_pca[:, 0], cmap="viridis", alpha=0.6
    )
    ax1.set_title("Original 3D data (colored by PC1)")
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.set_zlabel("Feature 3")
    plt.colorbar(scatter1, ax=ax1, label="PC1 value")

    ax2 = fig.add_subplot(122)
    scatter2 = ax2.scatter(
        X_pca[:, 0], X_pca[:, 1], c=X_pca[:, 0], cmap="viridis", alpha=0.6
    )
    ax2.set_title("2D PCA projection")
    ax2.set_xlabel("PCA Component 1")
    ax2.set_ylabel("PCA Component 2")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    pca.plot_pca_components(X, X_pca, n_components=2)
    pca.plot_scree()
    pca.plot_component_heatmap()
    pca.plot_reconstruction_error(X, max_components=2)
