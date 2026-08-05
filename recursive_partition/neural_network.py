"""Scikit-learn neural-network adaptors."""

from __future__ import annotations

import inspect

from sklearn.neural_network import MLPClassifier


class MLPClassifierAdapter(MLPClassifier):
    """Scikit-learn MLP classifier adaptor for recursive partitioning.

    ``RecursivePartitionClassifier`` only requires a base estimator to expose
    ``fit`` and ``predict``.  ``MLPClassifier`` already satisfies that
    contract, but its ``fit`` signature differs between supported
    scikit-learn versions: newer versions accept ``sample_weight`` while
    older versions do not.  This adaptor gives the recursive partitioner a
    stable ``fit(X, y, sample_weight=None)`` entry point and forwards weights
    when the installed scikit-learn supports them.

    All constructor parameters are inherited unchanged from
    :class:`sklearn.neural_network.MLPClassifier`, so cloning, pipelines,
    grid searches, and nested parameter access continue to work normally.
    The adaptor does not change the MLP algorithm or its hyperparameters.

    Parameters
    ----------
    **kwargs
        Any parameters accepted by ``MLPClassifier``.

    Notes
    -----
    On scikit-learn versions whose ``MLPClassifier.fit`` does not accept
    ``sample_weight``, passing weights raises a clear ``TypeError``.  Use the
    recursive classifier's ``sample_weight_policy="ignore"`` if weights are
    optional for a particular application.
    """

    def fit(self, X, y, sample_weight=None):
        """Fit the neural network, forwarding sample weights when supported."""

        if sample_weight is None:
            # Do not pass the keyword on older scikit-learn releases.
            return super().fit(X, y)

        parameters = inspect.signature(MLPClassifier.fit).parameters
        if "sample_weight" not in parameters:
            raise TypeError(
                "This scikit-learn version's MLPClassifier does not support "
                "sample_weight."
            )
        return super().fit(X, y, sample_weight=sample_weight)

    @staticmethod
    def _fit_accepts_sample_weight():
        """Report weight support for the recursive classifier's capability check."""

        return "sample_weight" in inspect.signature(MLPClassifier.fit).parameters


# Keep the spelling used in the public request available as well as the
# conventional American spelling used by most Python APIs.
MLPClassifierAdaptor = MLPClassifierAdapter


__all__ = ["MLPClassifierAdapter", "MLPClassifierAdaptor"]
