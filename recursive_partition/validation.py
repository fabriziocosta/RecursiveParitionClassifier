"""Validation and estimator capability helpers."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
from sklearn.utils.multiclass import check_classification_targets, type_of_target, unique_labels


def validate_classification_targets(y: Any) -> np.ndarray:
    """Validate binary or multiclass targets and return ordered labels."""

    check_classification_targets(y)
    if type_of_target(y) not in {"binary", "multiclass"}:
        raise ValueError("Recursive partition classifiers require binary or multiclass targets.")
    classes = unique_labels(y)
    if len(classes) < 2:
        raise ValueError("Recursive partition classifiers require at least two classes.")
    return classes


def validate_binary_targets(y: Any) -> np.ndarray:
    """Backward-compatible validator for callers that explicitly need binary labels."""

    classes = validate_classification_targets(y)
    if len(classes) != 2:
        raise ValueError("This operation requires exactly two classes.")
    return classes


def supports_sample_weight(estimator: Any) -> bool:
    """Detect whether an estimator's fit method accepts sample_weight."""

    declared_support = getattr(estimator, "_fit_accepts_sample_weight", None)
    if callable(declared_support):
        return bool(declared_support())

    try:
        signature = inspect.signature(estimator.fit)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    return "sample_weight" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


def validate_sample_weight(sample_weight: Any, n_samples: int) -> np.ndarray:
    weights = np.asarray(sample_weight, dtype=float)
    if weights.ndim != 1 or len(weights) != n_samples:
        raise ValueError("sample_weight must be a one-dimensional array with one value per sample.")
    if not np.all(np.isfinite(weights)):
        raise ValueError("sample_weight must contain only finite values.")
    if np.any(weights < 0):
        raise ValueError("sample_weight must be non-negative.")
    if not np.any(weights > 0):
        raise ValueError("sample_weight must contain at least one positive value.")
    return weights
