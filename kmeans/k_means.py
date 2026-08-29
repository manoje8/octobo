import numpy as np
import matplotlib.pyplot as plt

def kmeans(X, k, n_iters=100, tol=1e-4):
    n_samples, n_features = X.shape

    rng = np.random.default_rng(42)
    centroids = X[rng.choice(n_samples, k, replace=False)]


    for _ in range(n_iters):
        distances = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return labels, centroids

if __name__ == "__main__":
    X = np.random.randn(300, 2)
    X[:100] += [2, 2]
    X[100:200] += [-2, 2]
    X[200:] +=[0, -2]
    labels, centroids = kmeans(X, 3)

    # Plot results
    plt.figure(figsize=(10, 8))

    colors = ['red', 'blue', 'green']
    for i in range(3):
        mask = labels == i
        plt.scatter(X[mask, 0], X[mask, 1],
                    c=colors[i], label=f'Cluster {i + 1}',
                    alpha=0.6, s=50)

    plt.scatter(centroids[:, 0], centroids[:, 1],
                c='black', marker='X', s=200,
                edgecolors='white', linewidth=2,
                label='Centroids')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('K-means Clustering Results (k=3)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
