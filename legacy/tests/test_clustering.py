"""
Tests for the ManualKMeans implementation (src/clustering.py).

These verify correctness of the custom NumPy K-Means — the core algorithm
implemented from scratch to satisfy the course rubric requirement.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.clustering import ManualKMeans


def test_kmeans_separates_distinct_clusters():
    """K-Means must correctly label clearly separated clusters."""
    np.random.seed(0)
    cluster_a = np.random.randn(50, 2) + np.array([10.0, 10.0])
    cluster_b = np.random.randn(50, 2) + np.array([-10.0, -10.0])
    X = np.vstack([cluster_a, cluster_b])

    model = ManualKMeans(n_clusters=2, random_state=42)
    model.fit(X)

    labels_a = set(model.labels[:50])
    labels_b = set(model.labels[50:])

    assert len(labels_a) == 1, "All cluster-A points should share one label"
    assert len(labels_b) == 1, "All cluster-B points should share one label"
    assert labels_a != labels_b, "Two clusters must have different labels"


def test_inertia_decreases_with_more_clusters():
    """More clusters should never increase inertia (tighter fit with more centers)."""
    np.random.seed(42)
    X = np.random.randn(200, 5)

    prev_inertia = float('inf')
    for k in [2, 5, 10]:
        model = ManualKMeans(n_clusters=k, random_state=42)
        model.fit(X)
        assert model.inertia_ <= prev_inertia + 1e-6, f"Inertia rose going to k={k}"
        prev_inertia = model.inertia_


def test_labels_length_matches_input():
    """Number of labels must equal number of input samples."""
    X = np.random.randn(80, 10)
    model = ManualKMeans(n_clusters=5, random_state=0)
    model.fit(X)

    assert len(model.labels) == 80
    assert set(model.labels).issubset(set(range(5))), "Labels must be valid cluster indices"


def test_centroids_shape():
    """Centroids must have shape (k, n_features)."""
    X = np.random.randn(60, 8)
    model = ManualKMeans(n_clusters=3, random_state=0)
    model.fit(X)

    assert model.centroids.shape == (3, 8)


def test_silhouette_score_in_valid_range():
    """Silhouette score must lie in [-1, 1]."""
    np.random.seed(0)
    cluster_a = np.random.randn(30, 2) + np.array([5.0, 5.0])
    cluster_b = np.random.randn(30, 2) + np.array([-5.0, -5.0])
    X = np.vstack([cluster_a, cluster_b])

    model = ManualKMeans(n_clusters=2, random_state=42)
    model.fit(X)
    score = model.calculate_silhouette(X)

    assert -1.0 <= score <= 1.0, f"Silhouette score {score} out of valid range"


def test_silhouette_high_for_well_separated_clusters():
    """Silhouette score should be > 0.5 for clearly distinct clusters."""
    np.random.seed(1)
    cluster_a = np.random.randn(40, 2) * 0.3 + np.array([8.0, 0.0])
    cluster_b = np.random.randn(40, 2) * 0.3 + np.array([-8.0, 0.0])
    X = np.vstack([cluster_a, cluster_b])

    model = ManualKMeans(n_clusters=2, random_state=42)
    model.fit(X)
    score = model.calculate_silhouette(X)

    assert score > 0.5, f"Expected high silhouette for well-separated data, got {score:.3f}"


if __name__ == '__main__':
    test_kmeans_separates_distinct_clusters()
    test_inertia_decreases_with_more_clusters()
    test_labels_length_matches_input()
    test_centroids_shape()
    test_silhouette_score_in_valid_range()
    test_silhouette_high_for_well_separated_clusters()
    print("All clustering tests passed.")
