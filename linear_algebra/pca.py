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

      explained_variance = (S ** 2) / (n_samples - 1)
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
      return X_pca @ self.components_ +  self.mean



if __name__ == "__main__":
   rng = np.random.default_rng(42)
   n_samples = 500

   z1 = rng.normal(0, 3, n_samples)
   z2 = rng.normal(0, 1, n_samples)
   noise = rng.normal(0, 0.5, (n_samples, 3))

   basic = np.array([
      [1.0, 1.0, 0.0],
      [0.0, 1.0, 1.0]
   ])

   X = np.outer(z1, basic[0]) + np.outer(z2, basic[1]) + noise

   pca = PCA(n_components=2)
   X_pca = pca.fit_transform(X)

   print("Explained variance ratio: ", pca.explained_variance_ratio_)
   print("Sum of first 2 components' ratio: ", pca.explained_variance_ratio_.sum())
   print("Component shape: ", pca.components_.shape)
   print("Projected data shape: ", X_pca.shape)

   X_reconstructed = pca.inverse_transform(X_pca)
   reconstruction_mse = np.mean((X - X_reconstructed)**2)

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
      print(f"component {i} alignment in eigen composition: {cos_sim:.6f}"
            f"(1.0 = identical direction)"
            )