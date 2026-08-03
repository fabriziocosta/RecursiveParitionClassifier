"""QDA variants used by recursive partitioning."""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class EqualPriorQDA(QuadraticDiscriminantAnalysis):
    """Quadratic discriminant analysis with uniform local class priors.

    The recursive partitioner fits a fresh estimator at every node.  A node
    can contain a different number of classes from its parent, so a fixed
    ``priors=[0.5, 0.5]`` configuration is not suitable for multiclass data.
    This wrapper replaces ``priors`` immediately before each fit with a
    uniform vector whose length matches the classes present in that node.

    All constructor parameters are inherited from scikit-learn's
    :class:`~sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis`.
    """

    def fit(self, X, y):
        labels = np.unique(y)
        if labels.size < 2:
            raise ValueError("EqualPriorQDA requires at least two classes.")
        self.priors = np.full(labels.size, 1.0 / labels.size, dtype=float)
        return super().fit(X, y)
