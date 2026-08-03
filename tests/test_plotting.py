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
    figure = plot_probability_heatmap(model, X, y, grid_size=20)
    assert len(figure.axes) == 2
    assert np.isfinite(model.predict_proba(X)).all()
    plt.close(figure)
