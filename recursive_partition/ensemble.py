"""Parallel bagging ensemble for recursive partition classifiers."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.utils.validation import check_is_fitted

from .tree import RecursivePartitionClassifier
from .validation import validate_binary_targets, validate_sample_weight


def _fit_member(template, X, y, indices, seed, sample_weight):
    model = clone(template)
    if sample_weight is None:
        model.fit(X[indices], y[indices])
    else:
        model.fit(X[indices], y[indices], sample_weight=sample_weight[indices])
    return model, indices, seed


class BaggedRecursivePartitionClassifier(ClassifierMixin, BaseEstimator):
    """An independently fitted, mean-probability bagging ensemble."""

    def __init__(
        self,
        estimator=None,
        n_estimators=30,
        max_samples=1.0,
        bootstrap=True,
        n_jobs=None,
        random_state=None,
        aggregation="mean_proba",
        oob_score=False,
        verbose=0,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.aggregation = aggregation
        self.oob_score = oob_score
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse=["csr", "csc"], dtype=None)
        self.classes_ = validate_binary_targets(y)
        self.n_features_in_ = X.shape[1]
        if sample_weight is not None:
            sample_weight = validate_sample_weight(sample_weight, X.shape[0])
        self._validate_options(X.shape[0])
        template = self.estimator if self.estimator is not None else RecursivePartitionClassifier()
        if not callable(getattr(template, "fit", None)) or not callable(getattr(template, "predict_proba", None)):
            raise TypeError("estimator must implement fit and predict_proba methods.")
        self.estimator_ = clone(template)

        n_samples = X.shape[0]
        sample_size = self._sample_size(n_samples)
        rng = check_random_state(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=self.n_estimators, dtype=np.int64)
        samples = [self._draw_indices(rng, y, sample_size) for _ in range(self.n_estimators)]
        fitted = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_member)(template, X, y, indices, int(seed), sample_weight)
            for indices, seed in zip(samples, seeds)
        )
        self.estimators_ = [item[0] for item in fitted]
        self.estimators_samples_ = [item[1] for item in fitted]
        self.estimator_seeds_ = np.asarray(seeds, dtype=np.int64)
        self.oob_indices_ = [self._oob_indices(indices, n_samples) for indices in self.estimators_samples_]
        if self.oob_score:
            self._compute_oob_score(X, y)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X, accept_sparse=["csr", "csc"], dtype=None)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, expected {self.n_features_in_}.")
        probabilities = Parallel(n_jobs=self.n_jobs if self.n_estimators > 2 else 1)(
            delayed(_member_proba)(estimator, X) for estimator in self.estimators_
        )
        result = np.mean(np.asarray(probabilities), axis=0)
        result = np.maximum(result, 0.0)
        return result / result.sum(axis=1, keepdims=True)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]

    def _validate_options(self, n_samples):
        if not isinstance(self.n_estimators, (int, np.integer)) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if self.aggregation != "mean_proba":
            raise ValueError("aggregation must be 'mean_proba'.")
        if isinstance(self.max_samples, (float, np.floating)):
            if not 0 < self.max_samples <= 1:
                raise ValueError("float max_samples must be in (0, 1].")
        elif isinstance(self.max_samples, (int, np.integer)):
            if not 1 <= self.max_samples <= n_samples:
                raise ValueError("integer max_samples must be between 1 and n_samples.")
        else:
            raise TypeError("max_samples must be an integer or a float in (0, 1].")

    def _sample_size(self, n_samples):
        if isinstance(self.max_samples, (float, np.floating)):
            return max(1, int(np.ceil(float(self.max_samples) * n_samples)))
        return int(self.max_samples)

    def _draw_indices(self, rng, y, sample_size):
        classes = self.classes_
        if sample_size < 2:
            raise ValueError("At least two samples are required to ensure both classes in each member.")
        if self.bootstrap:
            indices = rng.randint(len(y), size=sample_size)
            if not np.any(y[indices] == classes[0]):
                indices[0] = rng.choice(np.flatnonzero(y == classes[0]))
            if not np.any(y[indices] == classes[1]):
                replace_at = 1 if indices[0] != indices[1] or sample_size > 2 else 0
                indices[replace_at] = rng.choice(np.flatnonzero(y == classes[1]))
            return np.asarray(indices, dtype=int)
        if sample_size < 2:
            raise ValueError("At least two samples are required for a binary no-replacement sample.")
        first = rng.choice(np.flatnonzero(y == classes[0]))
        second = rng.choice(np.flatnonzero(y == classes[1]))
        remaining_pool = np.setdiff1d(np.arange(len(y)), np.asarray([first, second]), assume_unique=False)
        if sample_size > len(remaining_pool) + 2:
            raise ValueError("max_samples is too large for sampling without replacement.")
        remainder = rng.choice(remaining_pool, size=sample_size - 2, replace=False)
        indices = np.concatenate(([first, second], remainder))
        rng.shuffle(indices)
        return indices.astype(int)

    @staticmethod
    def _oob_indices(indices, n_samples):
        in_bag = np.zeros(n_samples, dtype=bool)
        in_bag[indices] = True
        return np.flatnonzero(~in_bag)

    def _compute_oob_score(self, X, y):
        sums = np.zeros((len(y), 2), dtype=float)
        counts = np.zeros(len(y), dtype=int)
        for estimator, indices in zip(self.estimators_, self.oob_indices_):
            if len(indices) == 0:
                continue
            sums[indices] += estimator.predict_proba(X[indices])
            counts[indices] += 1
        decision = np.full_like(sums, np.nan)
        valid = counts > 0
        decision[valid] = sums[valid] / counts[valid, None]
        self.oob_decision_function_ = decision
        self.oob_score_ = float(np.mean(self.classes_[np.argmax(decision[valid], axis=1)] == y[valid])) if np.any(valid) else np.nan
        self.oob_counts_ = counts


def _member_proba(estimator, X):
    result = np.asarray(estimator.predict_proba(X), dtype=float)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError("Each ensemble estimator must return a two-column predict_proba result.")
    return result
