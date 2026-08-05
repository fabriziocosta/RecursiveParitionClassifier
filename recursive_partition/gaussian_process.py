"""Scikit-learn Gaussian-process classifier adaptors."""

from __future__ import annotations

import numpy as np

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from ._balancing import local_sample_weights, weighted_resample_indices


_DEFAULT_KERNEL = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)


class GaussianProcessClassifierAdapter(GaussianProcessClassifier):
    """Balanced Gaussian-process classifier for recursive partitioning.

    The default kernel is
    ``ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)``.  Gaussian
    process classifiers do not expose ``class_weight`` or ``sample_weight``
    in scikit-learn's public ``fit`` method, so this adaptor applies local
    class weights through deterministic weighted resampling before each fit.
    This is important for recursive partitioning: a child node can be heavily
    imbalanced even when the original training set is balanced.

    All Gaussian-process constructor parameters are retained, and
    ``class_weight`` is added as an adaptor parameter.  The estimator remains
    compatible with cloning, pipelines, grid searches, and nested parameter
    access.

    Parameters
    ----------
    kernel : kernel instance, default=ConstantKernel(1.0) * Matern(...)
        Kernel used by the Gaussian process classifier.
    class_weight : None, "balanced", or dict, default="balanced"
        Local class-balancing strategy.  ``None`` disables balancing.  The
        ``"balanced"`` option uses inverse-frequency class weights, while a
        dictionary supplies custom per-class weights.
    """

    def __init__(
        self,
        kernel=_DEFAULT_KERNEL,
        *,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        max_iter_predict=100,
        warm_start=False,
        copy_X_train=True,
        random_state=None,
        multi_class="one_vs_rest",
        n_jobs=None,
        class_weight="balanced",
    ):
        super().__init__(
            kernel=kernel,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            max_iter_predict=max_iter_predict,
            warm_start=warm_start,
            copy_X_train=copy_X_train,
            random_state=random_state,
            multi_class=multi_class,
            n_jobs=n_jobs,
        )
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        """Fit a locally balanced Gaussian-process classifier."""

        labels = np.asarray(y).reshape(-1)
        if len(labels) == 0:
            return super().fit(X, y)
        if self.class_weight is None and sample_weight is None:
            return super().fit(X, y)

        labels, combined_weights = local_sample_weights(
            labels, self.class_weight, sample_weight
        )
        indices = weighted_resample_indices(
            labels,
            combined_weights,
            self.random_state,
            balance_classes=self.class_weight is not None,
        )
        X_resampled = X.iloc[indices] if hasattr(X, "iloc") else X[indices]
        return super().fit(X_resampled, labels[indices])


# Keep the spelling used in the public request available as well as the
# conventional American spelling used by most Python APIs.
GaussianProcessClassifierAdaptor = GaussianProcessClassifierAdapter


__all__ = [
    "GaussianProcessClassifierAdapter",
    "GaussianProcessClassifierAdaptor",
]
