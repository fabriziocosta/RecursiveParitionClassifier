"""Compact internal tree node representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class _Node:
    node_id: int
    depth: int
    n_samples: int
    class_counts: np.ndarray
    predicted_class_index: int
    estimator: Optional[Any] = None
    negative_child: Optional["_Node"] = None
    positive_child: Optional["_Node"] = None
    is_leaf: bool = True

    @property
    def leaf_probability(self) -> np.ndarray:
        """Return the smoothed probability vector assigned by the owning tree."""

        raise AttributeError("leaf_probability is computed by the owning classifier")

