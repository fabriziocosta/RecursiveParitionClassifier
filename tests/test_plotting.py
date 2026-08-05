import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from recursive_partition import (
    RecursivePartitionClassifier,
    make_2d_dataset,
    plot_probability_heatmap,
)


def test_multiclass_probability_heatmap():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, y = make_2d_dataset("blobs", n_samples=90, n_classes=3, random_state=4)
    model = RecursivePartitionClassifier(
        base_estimator=LogisticRegression(max_iter=500),
        max_depth=3,
    ).fit(X, y)
    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_probability_heatmap(model, X, y, ax=axes[0], colorbar=False, grid_size=20)
    plot_probability_heatmap(model, X, y, ax=axes[1], colorbar=True, grid_size=20)
    assert len(figure.axes) == 3
    assert np.isfinite(model.predict_proba(X)).all()
    plt.close(figure)


def test_binary_probability_heatmap_includes_confidence_boundary():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC

    X, y = make_2d_dataset("moon", n_samples=120, random_state=4)
    model = RecursivePartitionClassifier(
        base_estimator=SVC(kernel="linear", C=1.0),
        max_depth=3,
    ).fit(X, y)
    figure = plot_probability_heatmap(
        model, X, y, colorbar=False, grid_size=24, decision_margin=0.1
    )
    contour_linewidths = [
        linewidth
        for collection in figure.axes[0].collections
        if hasattr(collection, "get_linewidth")
        for linewidth in collection.get_linewidth()
    ]
    assert any(np.isclose(linewidth, 2.0) for linewidth in contour_linewidths)
    assert any(np.isclose(linewidth, 0.45) for linewidth in contour_linewidths)
    plt.close(figure)
