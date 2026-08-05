"""Scikit-learn neural-network adaptors."""

from __future__ import annotations

import inspect

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import check_random_state

from .validation import validate_sample_weight


class MLPClassifierAdapter(MLPClassifier):
    """Scikit-learn MLP classifier adaptor for recursive partitioning.

    ``RecursivePartitionClassifier`` only requires a base estimator to expose
    ``fit`` and ``predict``.  ``MLPClassifier`` already satisfies that
    contract, but recursive nodes can become strongly imbalanced after a
    parent MLP routes observations.  This adaptor balances the classes in
    every local fit by default, preventing a child network from collapsing to
    its majority class.

    Its ``fit`` signature also differs between supported scikit-learn
    versions: newer versions accept ``sample_weight`` while older versions do
    not.  Native weights are used when available; otherwise the adaptor uses
    deterministic weighted resampling.

    All constructor parameters are inherited unchanged from
    :class:`sklearn.neural_network.MLPClassifier`, so cloning, pipelines,
    grid searches, and nested parameter access continue to work normally.
    ``class_weight=None`` disables the local balancing behavior.  The
    ``"balanced"`` setting follows scikit-learn's inverse-frequency class
    weighting convention.  A dictionary of per-class weights is also
    accepted.

    Parameters
    ----------
    class_weight : None, "balanced", or dict, default="balanced"
        Local class-balancing strategy.  This is an adaptor parameter; it is
        not a parameter of scikit-learn's native ``MLPClassifier``.

    Notes
    -----
    On scikit-learn versions whose ``MLPClassifier.fit`` does not accept
    ``sample_weight``, weights are applied through resampling instead.
    """

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        *,
        solver="adam",
        alpha=0.0001,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        random_state=None,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
        class_weight="balanced",
    ):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            power_t=power_t,
            max_iter=max_iter,
            shuffle=shuffle,
            random_state=random_state,
            tol=tol,
            verbose=verbose,
            warm_start=warm_start,
            momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            n_iter_no_change=n_iter_no_change,
            max_fun=max_fun,
        )
        self.class_weight = class_weight

    def fit(self, X, y, sample_weight=None):
        """Fit a locally class-balanced neural network."""

        labels = np.asarray(y).reshape(-1)
        if len(labels) == 0:
            return super().fit(X, y)

        if sample_weight is not None:
            sample_weight = validate_sample_weight(sample_weight, len(labels))
        balancing_weights = self._class_weights(labels)
        combined_weights = balancing_weights
        if sample_weight is not None:
            combined_weights = combined_weights * sample_weight

        if not np.any(combined_weights > 0):
            raise ValueError("class balancing and sample_weight produced no positive weights.")

        if self.class_weight is None and sample_weight is None:
            # Do not pass the keyword on older scikit-learn releases.
            return super().fit(X, y)

        if self._fit_accepts_sample_weight():
            return super().fit(X, y, sample_weight=combined_weights)

        indices = self._weighted_resample_indices(labels, combined_weights)
        X_resampled = X.iloc[indices] if hasattr(X, "iloc") else X[indices]
        return super().fit(X_resampled, labels[indices])

    def _class_weights(self, y):
        if self.class_weight is None:
            return np.ones(len(y), dtype=float)
        return np.asarray(compute_sample_weight(self.class_weight, y), dtype=float)

    def _weighted_resample_indices(self, y, weights):
        """Create equal-total-weight class samples for old sklearn releases."""

        classes = np.unique(y)
        class_totals = np.asarray(
            [weights[self._matches_label(y, value)].sum() for value in classes],
            dtype=float,
        )
        target_size = max(1, int(np.ceil(class_totals.max())))
        rng = check_random_state(self.random_state)
        resampled = []
        for class_value in classes:
            class_indices = np.flatnonzero(self._matches_label(y, class_value))
            class_weights = weights[class_indices]
            if not np.any(class_weights > 0):
                continue
            probabilities = class_weights / class_weights.sum()
            resampled.append(
                rng.choice(class_indices, size=target_size, replace=True, p=probabilities)
            )
        if not resampled:
            raise ValueError("class balancing produced no samples.")
        indices = np.concatenate(resampled)
        rng.shuffle(indices)
        return indices.astype(int)

    @staticmethod
    def _matches_label(values, label):
        try:
            return np.asarray(values == label, dtype=bool)
        except (TypeError, ValueError):
            return np.fromiter((value == label for value in values), dtype=bool, count=len(values))

    @staticmethod
    def _fit_accepts_sample_weight():
        """Report weight support for the recursive classifier's capability check."""

        return "sample_weight" in inspect.signature(MLPClassifier.fit).parameters


# Keep the spelling used in the public request available as well as the
# conventional American spelling used by most Python APIs.
MLPClassifierAdaptor = MLPClassifierAdapter


__all__ = ["MLPClassifierAdapter", "MLPClassifierAdaptor"]
