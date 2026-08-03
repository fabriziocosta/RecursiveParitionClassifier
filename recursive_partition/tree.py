"""Recursive classifier-driven partitioning tree."""

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.svm import SVC
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted

from ._node import _Node
from .validation import supports_sample_weight, validate_binary_targets, validate_sample_weight


class RecursivePartitionClassifier(ClassifierMixin, BaseEstimator):
    """A classifier-driven recursive top-down binary partitioner.

    Each non-leaf node fits a fresh clone of ``base_estimator`` on the true
    labels belonging to that node. The estimator's predictions define the two
    child subsets; this is not a CART impurity-optimized split.

    Parameters are intentionally assigned unchanged in ``__init__`` so that
    cloning, pipelines, and parameter searches follow scikit-learn conventions.
    """

    def __init__(
        self,
        base_estimator=None,
        probability_mode="leaf_frequency",
        probability_smoothing=1.0,
        on_fit_failure="leaf",
        node_fit_validator=None,
        max_depth=None,
        max_nodes=None,
        sample_weight_policy="raise",
    ):
        self.base_estimator = base_estimator
        self.probability_mode = probability_mode
        self.probability_smoothing = probability_smoothing
        self.on_fit_failure = on_fit_failure
        self.node_fit_validator = node_fit_validator
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.sample_weight_policy = sample_weight_policy

    def fit(self, X, y, sample_weight=None):
        """Fit the recursive partitioner and return ``self``."""

        X, y = check_X_y(X, y, accept_sparse=["csr", "csc"], dtype=None)
        self.classes_ = validate_binary_targets(y)
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        if sample_weight is not None:
            sample_weight = validate_sample_weight(sample_weight, X.shape[0])
        self._validate_options()

        estimator_template = self.base_estimator
        if estimator_template is None:
            estimator_template = SVC(kernel="linear", class_weight="balanced")
        self._validate_base_estimator(estimator_template)
        self.base_estimator_ = clone(estimator_template)

        self._X_is_sparse_ = sparse.issparse(X)
        self._n_features_ = X.shape[1]
        self.nodes_: List[_Node] = []
        root = self._make_node(0, 0, np.arange(X.shape[0], dtype=int), y, sample_weight)
        self.tree_ = root

        # Iterative construction protects users from Python recursion limits.
        stack: List[Tuple[_Node, np.ndarray]] = [(root, np.arange(X.shape[0], dtype=int))]
        while stack:
            node, indices = stack.pop()
            if self._must_stop(node, indices, y):
                continue

            if self.max_nodes is not None and len(self.nodes_) + 2 > self.max_nodes:
                continue

            X_node = X[indices]
            y_node = y[indices]
            try:
                estimator = clone(estimator_template)
                if self.node_fit_validator is not None and not self.node_fit_validator(
                    estimator, X_node, y_node
                ):
                    continue
                self._fit_one(estimator, X_node, y_node, None if sample_weight is None else sample_weight[indices])
                predicted = np.asarray(estimator.predict(X_node)).reshape(-1)
                if predicted.shape[0] != indices.shape[0]:
                    raise ValueError("base_estimator.predict returned the wrong number of labels")
                negative = self._matches_label(predicted, self.classes_[0])
                positive = self._matches_label(predicted, self.classes_[1])
                if np.any(~(negative | positive)):
                    raise ValueError("base_estimator.predict returned an unknown class label")
                if not np.any(negative) or not np.any(positive):
                    continue
            except Exception as exc:
                self._handle_fit_failure(exc, node)
                continue

            node.estimator = estimator
            node.is_leaf = False
            negative_indices = indices[negative]
            positive_indices = indices[positive]
            node.negative_child = self._make_node(
                len(self.nodes_), node.depth + 1, negative_indices, y, sample_weight
            )
            node.positive_child = self._make_node(
                len(self.nodes_), node.depth + 1, positive_indices, y, sample_weight
            )
            stack.append((node.positive_child, positive_indices))
            stack.append((node.negative_child, negative_indices))

        self.n_nodes_ = len(self.nodes_)
        self.max_depth_ = max(node.depth for node in self.nodes_)
        self.n_leaves_ = sum(node.is_leaf for node in self.nodes_)
        return self

    def predict(self, X):
        """Predict the majority true class in each reached terminal leaf."""

        leaves, _, _ = self._traverse(X)
        return self.classes_[np.fromiter((node.predicted_class_index for node in leaves), dtype=int)]

    def predict_proba(self, X):
        """Return two-class probabilities using leaf frequencies or routing probabilities."""

        leaves, base_probabilities, sample_leaf_ids = self._traverse(
            X, collect_base_probabilities=self.probability_mode == "base_estimator"
        )
        probabilities = np.vstack([self._leaf_probability(node) for node in leaves])
        if self.probability_mode == "base_estimator":
            available = np.all(np.isfinite(base_probabilities), axis=1)
            probabilities[available] = base_probabilities[available]
        probabilities = np.asarray(probabilities, dtype=float)
        probabilities[probabilities < 0] = 0.0
        normalizers = probabilities.sum(axis=1)
        invalid = ~np.isfinite(normalizers) | (normalizers <= 0)
        probabilities[~invalid] /= normalizers[~invalid, None]
        if np.any(invalid):
            probabilities[invalid] = 0.5
        return probabilities

    def apply(self, X):
        """Return the terminal node ID reached by each sample."""

        _, _, sample_leaf_ids = self._traverse(X)
        return sample_leaf_ids

    def decision_path(self, X):
        """Return a CSR indicator matrix of nodes visited by each sample."""

        _, _, _, rows, cols = self._traverse(X, return_path=True)
        data = np.ones(len(rows), dtype=np.int8)
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(check_array(X, accept_sparse=["csr", "csc"])), self.n_nodes_))

    def get_depth(self):
        """Return the maximum root-to-leaf depth, with the root at depth zero."""

        check_is_fitted(self, "tree_")
        return self.max_depth_

    def get_n_leaves(self):
        """Return the number of terminal nodes."""

        check_is_fitted(self, "tree_")
        return self.n_leaves_

    def _validate_options(self):
        if self.probability_mode not in ("leaf_frequency", "base_estimator"):
            raise ValueError("probability_mode must be 'leaf_frequency' or 'base_estimator'.")
        if not isinstance(self.probability_smoothing, (int, float)) or self.probability_smoothing < 0:
            raise ValueError("probability_smoothing must be a non-negative number.")
        if self.on_fit_failure not in ("leaf", "raise"):
            raise ValueError("on_fit_failure must be 'leaf' or 'raise'.")
        if self.sample_weight_policy not in ("raise", "ignore"):
            raise ValueError("sample_weight_policy must be 'raise' or 'ignore'.")
        if self.max_depth is not None and (not isinstance(self.max_depth, (int, np.integer)) or self.max_depth < 0):
            raise ValueError("max_depth must be None or a non-negative integer.")
        if self.max_nodes is not None and (not isinstance(self.max_nodes, (int, np.integer)) or self.max_nodes < 1):
            raise ValueError("max_nodes must be None or a positive integer.")
        if self.node_fit_validator is not None and not callable(self.node_fit_validator):
            raise TypeError("node_fit_validator must be callable or None.")

    @staticmethod
    def _validate_base_estimator(estimator):
        if not callable(getattr(estimator, "fit", None)) or not callable(getattr(estimator, "predict", None)):
            raise TypeError("base_estimator must implement callable fit and predict methods.")

    def _fit_one(self, estimator, X, y, sample_weight):
        if sample_weight is not None and supports_sample_weight(estimator):
            estimator.fit(X, y, sample_weight=sample_weight)
        elif sample_weight is not None and self.sample_weight_policy == "raise":
            raise TypeError(
                f"{estimator.__class__.__name__}.fit does not accept sample_weight; "
                "set sample_weight_policy='ignore' to omit it."
            )
        else:
            estimator.fit(X, y)

    def _handle_fit_failure(self, exc, node):
        if self.on_fit_failure == "raise":
            raise RuntimeError(
                f"base_estimator failed while fitting recursive node {node.node_id}: {exc}"
            ) from exc

    def _make_node(self, node_id, depth, indices, y, sample_weight):
        if sample_weight is None:
            counts = np.bincount(
                np.searchsorted(self.classes_, y[indices]), minlength=2
            ).astype(float)
        else:
            counts = np.zeros(2, dtype=float)
            for class_index, class_value in enumerate(self.classes_):
                counts[class_index] = sample_weight[indices][self._matches_label(y[indices], class_value)].sum()
        predicted_class_index = int(np.argmax(counts))
        node = _Node(
            node_id=node_id,
            depth=depth,
            n_samples=len(indices),
            class_counts=counts,
            predicted_class_index=predicted_class_index,
        )
        self.nodes_.append(node)
        return node

    def _must_stop(self, node, indices, y):
        return (
            node.n_samples <= 1
            or np.all(self._matches_label(y[indices], y[indices][0]))
            or (self.max_depth is not None and node.depth >= self.max_depth)
        )

    def _matches_label(self, values, label):
        try:
            return np.asarray(values == label, dtype=bool)
        except (TypeError, ValueError):
            return np.fromiter((value == label for value in values), dtype=bool, count=len(values))

    def _leaf_probability(self, node):
        alpha = float(self.probability_smoothing)
        return (node.class_counts + alpha) / (node.class_counts.sum() + 2.0 * alpha)

    def _traverse(self, X, collect_base_probabilities=False, return_path=False):
        check_is_fitted(self, "tree_")
        X = check_array(X, accept_sparse=["csr", "csc"], dtype=None)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        n_samples = X.shape[0]
        leaves: List[Optional[_Node]] = [None] * n_samples
        leaf_ids = np.empty(n_samples, dtype=int)
        base_probabilities = np.full((n_samples, 2), np.nan, dtype=float)
        rows: List[int] = []
        cols: List[int] = []
        stack: List[Tuple[_Node, np.ndarray]] = [(self.tree_, np.arange(n_samples, dtype=int))]
        while stack:
            node, indices = stack.pop()
            if return_path:
                rows.extend(indices.tolist())
                cols.extend([node.node_id] * len(indices))
            if node.is_leaf:
                for index in indices:
                    leaves[index] = node
                leaf_ids[indices] = node.node_id
                continue
            predicted = np.asarray(node.estimator.predict(X[indices])).reshape(-1)
            if predicted.shape[0] != len(indices):
                raise ValueError("base_estimator.predict returned the wrong number of labels during prediction")
            negative = self._matches_label(predicted, self.classes_[0])
            positive = self._matches_label(predicted, self.classes_[1])
            if np.any(~(negative | positive)) or np.any(negative & positive):
                raise ValueError("base_estimator.predict returned an invalid class label during prediction")
            if collect_base_probabilities:
                base_probabilities[indices] = self._node_probabilities(node.estimator, X[indices], predicted)
            if np.any(negative):
                stack.append((node.negative_child, indices[negative]))
            if np.any(positive):
                stack.append((node.positive_child, indices[positive]))
        resolved_leaves = [node for node in leaves if node is not None]
        if len(resolved_leaves) != n_samples:
            raise RuntimeError("tree traversal did not assign every sample to a terminal node")
        if return_path:
            return resolved_leaves, base_probabilities, leaf_ids, rows, cols
        return resolved_leaves, base_probabilities, leaf_ids

    def _node_probabilities(self, estimator, X, predicted):
        probabilities = np.zeros((len(predicted), 2), dtype=float)
        if callable(getattr(estimator, "predict_proba", None)):
            try:
                raw = np.asarray(estimator.predict_proba(X), dtype=float)
                estimator_classes = getattr(estimator, "classes_", self.classes_)
                for column, label in enumerate(np.asarray(estimator_classes).reshape(-1)):
                    for global_index, global_label in enumerate(self.classes_):
                        if bool(self._matches_label(np.asarray([label], dtype=object), global_label)[0]):
                            if column < raw.shape[1]:
                                probabilities[:, global_index] = raw[:, column]
                            break
                if raw.ndim == 2 and raw.shape[0] == len(predicted) and np.all(np.isfinite(probabilities)):
                    return probabilities
            except Exception:
                pass
        probabilities[:, 0] = self._matches_label(predicted, self.classes_[0])
        probabilities[:, 1] = self._matches_label(predicted, self.classes_[1])
        return probabilities

