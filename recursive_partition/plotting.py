"""Optional plotting helpers for two-dimensional classifier demonstrations."""

from __future__ import annotations

from typing import Any

import numpy as np


def plot_probability_heatmap(
    model: Any,
    X: np.ndarray,
    y: np.ndarray | None = None,
    *,
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
    padding: float = 0.55,
    grid_size: int = 250,
    title: str | None = None,
):
    """Plot binary or multiclass probabilities over a 2D feature grid.

    Multiclass backgrounds use a probability-weighted ``tab10`` color mixture.
    Normalized entropy fades uncertain regions toward white, while black
    contours show predicted-class boundaries and the confidence boundary.
    Matplotlib is imported lazily so plotting remains an optional dependency.
    """

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must have shape (n_samples, 2) for heatmap plotting")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    X_points = X if X_train is None else np.asarray(X_train)
    y_points = y if y_train is None else np.asarray(y_train)
    if y_points is None:
        raise ValueError("y or y_train is required to plot data points")

    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - padding, X[:, 0].max() + padding, grid_size),
        np.linspace(X[:, 1].min() - padding, X[:, 1].max() + padding, grid_size),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    probabilities = np.asarray(model.predict_proba(grid), dtype=float).reshape(
        xx.shape + (len(model.classes_),)
    )
    classes = np.asarray(model.classes_)
    cmap = plt.get_cmap("tab10")
    palette = np.asarray([cmap(index % 10)[:3] for index in range(len(classes))])
    predicted_indices = np.argmax(probabilities, axis=2)

    if len(classes) > 2:
        color_mixture = probabilities @ palette
        safe_probabilities = np.clip(probabilities, np.finfo(float).tiny, 1.0)
        entropy = -np.sum(probabilities * np.log(safe_probabilities), axis=2)
        confidence = 1.0 - entropy / np.log(len(classes))
        rgb = 1.0 - confidence[..., None] * (1.0 - color_mixture)
        fig, axis = plt.subplots(figsize=(9, 6.5))
        axis.imshow(
            np.clip(rgb, 0.0, 1.0),
            origin="lower",
            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
            interpolation="nearest",
            aspect="equal",
        )
        boundaries = np.arange(0.5, len(classes) - 0.5, 1.0)
        axis.contour(xx, yy, predicted_indices, levels=boundaries,
                     linewidths=2, colors="black")
        if confidence.min() <= 0.5 <= confidence.max():
            axis.contour(xx, yy, confidence, levels=[0.5], linewidths=1.0,
                         colors="black")
        confidence_scale = ScalarMappable(norm=Normalize(0.0, 1.0), cmap="Greys_r")
        confidence_scale.set_array(confidence)
        colorbar = fig.colorbar(confidence_scale, ax=axis)
        colorbar.set_label("Confidence (1 − normalized entropy)")
    else:
        fig, axis = plt.subplots(figsize=(9, 6.5))
        probability = probabilities[:, :, 1]
        axis.contourf(xx, yy, probability, levels=np.linspace(0, 1, 41),
                      cmap="RdBu_r", vmin=0, vmax=1, alpha=0.9)
        axis.contour(xx, yy, probability, levels=[0.5], linewidths=1.0,
                     colors="black")

    for class_index, class_value in enumerate(classes):
        mask = y_points == class_value
        axis.scatter(
            X_points[mask, 0], X_points[mask, 1], s=24,
            color=palette[class_index], edgecolors="white", linewidths=0.45,
            label=f"Class {class_value}",
        )
    axis.set_title(title or "Class probability heatmap")
    axis.set_xlabel("Feature 1")
    axis.set_ylabel("Feature 2")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="upper right")
    fig.tight_layout()
    return fig
