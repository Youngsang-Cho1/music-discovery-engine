"""
Manual K-Means clustering implementation (Lloyd's algorithm + K-Means++ init).
Used for the "Auto Playlist" feature — grouping songs into mood clusters.
No sklearn dependency.
"""
import numpy as np


class ManualKMeans:
    """
    K-Means from scratch.

    Algorithm:
    1. K-Means++ initialization — choose initial centroids that are spread out
       (first centroid is random; each subsequent centroid is drawn with
       probability proportional to squared distance from the nearest existing
       centroid). This avoids the poor local minima that random init suffers from.

    2. Lloyd's algorithm — iterate:
       (a) Assign each point to its nearest centroid (Euclidean distance).
       (b) Update each centroid to the mean of points assigned to it.
       Stop when centroids stop moving (or max_iter reached).

    3. Multiple restarts — run n_init times with different seeds and keep the
       clustering with the lowest inertia (sum of squared distances to centroids).
       K-Means is sensitive to initialization; restarts mitigate that.
    """

    def __init__(self, n_clusters: int = 8, max_iter: int = 300,
                 n_init: int = 10, tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state

    def _init_centroids_pp(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """K-Means++ initialization."""
        n = X.shape[0] # Number of data points
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype) # Placeholder for K centroids

        # First centroid: pick one point uniformly at random from X
        centroids[0] = X[rng.integers(n)]

        # Subsequent centroids: choose points with probability proportional to squared distance
        for i in range(1, self.n_clusters):
            # Calculate squared Euclidean distance from every point to its closest already-chosen centroid
            # X[:, None, :] (n, 1, d) minus centroids[:i][None, :, :] (1, i, d) -> broadcasted distances
            dists_sq = np.min(
                np.sum((X[:, None, :] - centroids[:i][None, :, :]) ** 2, axis=2),
                axis=1,
            )
            total = dists_sq.sum() # Sum of squared distances for normalization
            if total == 0:
                # Fallback to random if all points overlap with chosen centroids
                centroids[i] = X[rng.integers(n)]
            else:
                probs = dists_sq / total # Probability distribution (closer points = lower prob)
                centroids[i] = X[rng.choice(n, p=probs)] # Weighted random selection

        return centroids

    def _lloyds(self, X: np.ndarray, centroids: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Lloyd's algorithm inner loop."""
        labels = np.zeros(X.shape[0], dtype=np.int64) # Buffer for cluster assignments

        for _ in range(self.max_iter):
            # Assignment: Compute distance from every point to every centroid
            dists_sq = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists_sq, axis=1) # Pick the index of the closest centroid

            # Update: New centroid = mean coordinates of all points assigned to it
            new_centroids = np.empty_like(centroids)
            for k in range(self.n_clusters):
                mask = new_labels == k # Points belonging to cluster k
                if mask.any():
                    new_centroids[k] = X[mask].mean(axis=0) # Take the geometric mean
                else:
                    new_centroids[k] = centroids[k] # Preserve old centroid if cluster is empty

            # Check convergence: total movement of centroids
            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids # Update centroids for next iteration
            labels = new_labels # Update labels
            if shift < self.tol: # Stop if shift is below tolerance
                break

        # Calculate final inertia: total intra-cluster sum of squares
        inertia = float(np.sum(np.sum((X - centroids[labels]) ** 2, axis=1)))
        return labels, centroids, inertia

    def fit(self, X: np.ndarray) -> "ManualKMeans":
        """Run K-Means with n_init restarts; keep the best by inertia."""
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.default_rng(self.random_state)

        best_labels = None
        best_centroids = None
        best_inertia = np.inf

        for i in range(self.n_init):
            # Each restart gets its own sub-rng for reproducibility
            sub_rng = np.random.default_rng(self.random_state + i)
            centroids = self._init_centroids_pp(X, sub_rng)
            labels, centroids, inertia = self._lloyds(X, centroids)

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels
                best_centroids = centroids

        self.labels_ = best_labels
        self.centroids_ = best_centroids
        self.inertia_ = best_inertia
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new points to the nearest learned centroid."""
        X = np.asarray(X, dtype=np.float64)
        dists_sq = np.sum((X[:, None, :] - self.centroids_[None, :, :]) ** 2, axis=2)
        return np.argmin(dists_sq, axis=1)


def manual_silhouette_score(X: np.ndarray, labels: np.ndarray, sample_size: int = 500) -> float:
    """
    Approximate silhouette score via random subsampling (no sklearn).

    For each sampled point i:
      a(i) = mean distance to other points in the same cluster
      b(i) = min mean distance to points in any other cluster
      s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Returns mean s over sampled points. Range [-1, 1]; higher = better.

    Reference ranges:
      Iris (4D, 3 clean classes) : ~0.55  (gold standard)
      8D metadata only           : ~0.20–0.35
      64D (meta + audio, K=2–10) : ~0.13–0.26  ← this dataset (observed)
      Pure random 64D noise      : ~0.00–0.05

    Why lower than Iris: "concentration of measure" — as d grows, pairwise
    distances concentrate around their mean, so a(i) ≈ b(i) → s(i) → 0.
    This dataset scores well above the random floor because genre structure
    is genuine, but still below Iris because 64D dilutes the signal.
    Any positive mean score indicates real cluster structure.
    """
    rng = np.random.default_rng(42)
    n = len(X)
    idx = rng.choice(n, size=min(sample_size, n), replace=False)
    X_s = X[idx].astype(np.float64)
    L_s = labels[idx]
    unique = np.unique(L_s)
    m = len(X_s)

    # Pairwise Euclidean distance matrix (m × m) — vectorized
    diff = X_s[:, None, :] - X_s[None, :, :]   # (m, m, d)
    D = np.sqrt(np.sum(diff ** 2, axis=2))       # (m, m)

    scores = np.empty(m)
    for i in range(m):
        c = L_s[i]
        same_mask = L_s == c
        same_mask[i] = False  # exclude self

        if not same_mask.any():
            scores[i] = 0.0
            continue

        a = D[i, same_mask].mean()

        b = np.inf
        for oc in unique:
            if oc == c:
                continue
            other_mask = L_s == oc
            if not other_mask.any():
                continue
            mean_d = D[i, other_mask].mean()
            if mean_d < b:
                b = mean_d

        if np.isinf(b):
            scores[i] = 0.0
        else:
            denom = max(a, b)
            scores[i] = (b - a) / denom if denom > 0 else 0.0

    return float(scores.mean())
