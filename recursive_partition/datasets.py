"""Small, self-contained two-dimensional classification datasets."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris, make_blobs, make_circles, make_moons
from sklearn.datasets import make_classification as sklearn_make_classification


DATASET_ALIASES = {
    "moon": "moon", "moons": "moon", "iris": "iris",
    "gaussian": "gaussian", "gaussians": "gaussian",
    "two_gaussians": "gaussian", "2 equal isotropic gaussians": "gaussian",
    "2_equal_isotropic_gaussians": "gaussian",
    "two_equal_isotropic_gaussians": "gaussian", "blob": "blobs",
    "blobs": "blobs", "gaussian_blobs": "blobs", "circle": "circles",
    "circles": "circles", "xor": "xor", "spiral": "spirals",
    "spirals": "spirals", "anisotropic": "anisotropic_blobs",
    "anisotropic_blobs": "anisotropic_blobs", "checker": "checkerboard",
    "checkerboard": "checkerboard", "classification": "classification",
    "make_classification": "classification",
}


def _canonical_dataset(dataset: str) -> str:
    try:
        return DATASET_ALIASES[dataset.lower()]
    except (AttributeError, KeyError) as error:
        choices = ", ".join(sorted(set(DATASET_ALIASES.values())))
        raise ValueError(f"dataset must be one of: {choices}") from error


def _make_equal_isotropic_gaussians(n_samples, mean_distance, standard_deviation, random_state):
    if n_samples < 2 or mean_distance <= 0 or standard_deviation <= 0:
        raise ValueError("n_samples must be at least 2 and Gaussian parameters must be positive")
    rng = np.random.default_rng(random_state)
    counts = (n_samples // 2, n_samples - n_samples // 2)
    covariance = np.eye(2) * standard_deviation**2
    centers = ((-mean_distance / 2, 0.0), (mean_distance / 2, 0.0))
    X = np.vstack([rng.multivariate_normal(center, covariance, size=count)
                   for center, count in zip(centers, counts)])
    y = np.repeat((0, 1), counts)
    order = rng.permutation(n_samples)
    return X[order], y[order]


def _make_blobs(n_samples, n_classes, center_radius, cluster_standard_deviation, random_state):
    if not isinstance(n_classes, (int, np.integer)) or isinstance(n_classes, bool) or n_classes < 2:
        raise ValueError("n_classes must be an integer of at least 2")
    if n_samples < n_classes or center_radius <= 0 or cluster_standard_deviation <= 0:
        raise ValueError("invalid blob parameters")
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_classes), endpoint=False)
    centers = center_radius * np.column_stack((np.cos(angles), np.sin(angles)))
    return make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=2,
        cluster_std=cluster_standard_deviation,
        random_state=random_state,
    )


def _make_xor(n_samples, cluster_standard_deviation, random_state):
    centers = np.array([(-2.0, -2.0), (-2.0, 2.0), (2.0, -2.0), (2.0, 2.0)])
    X, cluster_labels = make_blobs(n_samples=n_samples, centers=centers,
                                   cluster_std=cluster_standard_deviation,
                                   random_state=random_state)
    return X, np.asarray([0, 1, 1, 0])[cluster_labels]


def _make_spirals(n_samples, turns, noise, random_state):
    rng = np.random.default_rng(random_state)
    counts = (n_samples // 2, n_samples - n_samples // 2)
    theta = np.linspace(0.2, 2.0 * np.pi * turns, counts[0])
    radius = np.linspace(0.2, 1.0, counts[0])
    first = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    second = np.column_stack((radius * np.cos(theta + np.pi), radius * np.sin(theta + np.pi)))
    X = np.vstack((first, second)) + rng.normal(scale=noise, size=(n_samples, 2))
    y = np.repeat((0, 1), counts)
    order = rng.permutation(n_samples)
    return X[order], y[order]


def _make_checkerboard(n_samples, cells, extent, label_noise, random_state):
    rng = np.random.default_rng(random_state)
    X = rng.uniform(-extent, extent, size=(n_samples, 2))
    cell_indices = np.floor((X + extent) / (2.0 * extent) * cells).astype(int)
    y = (cell_indices[:, 0] + cell_indices[:, 1]) % 2
    flips = rng.random(n_samples) < label_noise
    return X, np.where(flips, 1 - y, y)


def make_2d_dataset(
    dataset="moon", *, n_samples=600, noise=0.24, random_state=7,
    feature_indices=(0, 1), mean_distance=3.0, standard_deviation=1.0,
    n_classes=3, blob_center_radius=3.0, blob_cluster_standard_deviation=1.5,
    circle_factor=0.5, spiral_turns=1.5, spiral_noise=0.12,
    checkerboard_cells=4, checkerboard_extent=4.0, checkerboard_label_noise=0.0,
    anisotropy=3.0, rotation=0.35, classification_class_sep=1.0,
    classification_flip_y=0.05,
):
    """Create one of the shared two-dimensional demo datasets."""

    name = _canonical_dataset(dataset)
    if name == "moon":
        return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    if name == "circles":
        return make_circles(n_samples=n_samples, noise=noise, factor=circle_factor,
                            random_state=random_state)
    if name == "iris":
        iris = load_iris()
        return iris.data[:, feature_indices], iris.target
    if name == "blobs":
        return _make_blobs(n_samples, n_classes, blob_center_radius,
                           blob_cluster_standard_deviation, random_state)
    if name == "xor":
        return _make_xor(n_samples, blob_cluster_standard_deviation, random_state)
    if name == "spirals":
        return _make_spirals(n_samples, spiral_turns, spiral_noise, random_state)
    if name == "checkerboard":
        return _make_checkerboard(n_samples, checkerboard_cells, checkerboard_extent,
                                  checkerboard_label_noise, random_state)
    if name == "classification":
        return sklearn_make_classification(
            n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0,
            n_repeated=0, n_classes=n_classes, n_clusters_per_class=1,
            class_sep=classification_class_sep, flip_y=classification_flip_y,
            random_state=random_state,
        )
    if name == "anisotropic_blobs":
        X, y = _make_blobs(n_samples, n_classes, blob_center_radius,
                           blob_cluster_standard_deviation, random_state)
        angle = float(rotation)
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        return X @ np.diag((anisotropy, 1.0)) @ rotation_matrix.T, y
    return _make_equal_isotropic_gaussians(n_samples, mean_distance,
                                           standard_deviation, random_state)
