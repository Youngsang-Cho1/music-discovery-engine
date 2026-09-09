"""
Manual PCA implementation using covariance matrix + SVD.
Used for 2D visualization of audio embeddings in the Streamlit Evaluation tab.
No sklearn dependency.
"""
import numpy as np


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize X to zero mean and unit variance (z-score normalization).
    This is critical before PCA: without it, high-variance features
    (e.g. tempo in BPM) would dominate the principal components purely
    because of their scale, not because they carry more information.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # avoid division by zero for constant features
    return (X - mean) / std, mean, std


def manual_pca(X: np.ndarray, n_components: int = 2):
    """
    Manual PCA via covariance matrix eigendecomposition using SVD.

    Steps:
    1. Center X (subtract mean) so the covariance is computed around the origin.
    2. Compute the covariance matrix: C = X^T @ X / (n-1).
       Each entry C[i,j] measures how much feature i and j vary together.
    3. Apply SVD to C: C = U S V^T.
       - U: eigenvectors (principal component directions)
       - S: eigenvalues (variance explained per component)
    4. Project X onto the top-k principal components: X_pca = X_centered @ U[:, :k]

    Returns (X_pca, explained_variance_ratio, components, mean).
      - X_pca: (n, n_components) — points projected into PC space
      - components: (d, n_components) — PC basis vectors (columns are PCs),
        reusable for projecting new points: Y = (new - mean) @ components
      - mean: (d,) — data mean used for centering (for projecting new points)
    """
    n = X.shape[0] # Step 0: Get the number of sample points

    mean_vec = X.mean(axis=0) # Step 1: Calculate the average value of each feature
    X_centered = X - mean_vec # Step 2: Center data by subtracting the mean from every point

    # Step 3: Compute the Covariance Matrix: (X^T @ X) / (n - 1)
    # This matrix captures how each pair of features varies together.
    cov = X_centered.T @ X_centered / (n - 1)

    # Step 4: Perform Singular Value Decomposition (SVD) on the Covariance matrix
    # U: Orthogonal matrix where columns are Principal Components (Eigenvectors)
    # S: Diagonal matrix of singular values (Variance explained / Eigenvalues)
    U, S, _ = np.linalg.svd(cov)

    components = U[:, :n_components] # Step 5: Select the top K Principal Components
    X_pca = X_centered @ components # Step 6: Project the original D-dimensional data onto K-dimensions
    explained_variance_ratio = S[:n_components] / S.sum() # Calculate the % of total variance captured

    return X_pca, explained_variance_ratio, components, mean_vec # Return results for visualization
