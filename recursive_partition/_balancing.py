"""Local class-balancing helpers for recursive base-estimator adaptors."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.class_weight import compute_sample_weight

from .validation import validate_sample_weight


def local_sample_weights(y: Any, class_weight, sample_weight=None):
    """Return flattened labels and combined local class/sample weights."""

    labels = np.asarray(y).reshape(-1)
    if sample_weight is not None:
        sample_weight = validate_sample_weight(sample_weight, len(labels))
    balancing_weights = (
        np.ones(len(labels), dtype=float)
        if class_weight is None
        else np.asarray(compute_sample_weight(class_weight, labels), dtype=float)
    )
    combined = balancing_weights if sample_weight is None else balancing_weights * sample_weight
    if len(labels) and not np.any(combined > 0):
        raise ValueError("class balancing and sample_weight produced no positive weights.")
    return labels, combined


def weighted_resample_indices(y, weights, random_state, balance_classes=True):
    """Resample weighted observations, optionally equalizing class totals."""

    rng = check_random_state(random_state)
    if not balance_classes:
        probabilities = weights / weights.sum()
        indices = rng.choice(len(y), size=len(y), replace=True, p=probabilities)
        rng.shuffle(indices)
        return indices.astype(int)

    classes = np.unique(y)
    class_totals = np.asarray(
        [weights[_matches_label(y, value)].sum() for value in classes],
        dtype=float,
    )
    target_size = max(1, int(np.ceil(class_totals.max())))
    resampled = []
    for class_value in classes:
        class_indices = np.flatnonzero(_matches_label(y, class_value))
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


def _matches_label(values, label):
    try:
        return np.asarray(values == label, dtype=bool)
    except (TypeError, ValueError):
        return np.fromiter((value == label for value in values), dtype=bool, count=len(values))
