#!/usr/bin/env python3
"""Compare recursive base classifiers and their bagged form on two moons."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from recursive_partition import BaggedRecursivePartitionClassifier, EqualPriorQDA, RecursivePartitionClassifier


def plot_model(ax, name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
    y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 240), np.linspace(y_min, y_max, 240))
    grid = np.c_[xx.ravel(), yy.ravel()]
    proba = model.predict_proba(grid)[:, 1].reshape(xx.shape)
    accuracy = model.score(X_test, y_test)
    ax.contourf(xx, yy, proba, levels=30, cmap="RdBu_r", alpha=0.55)
    ax.contour(xx, yy, proba, levels=[0.5], colors="black", linewidths=1.2)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="RdBu_r", edgecolors="white", linewidths=0.4, s=22)
    ax.set_title(f"{name}\nTest accuracy: {accuracy:.3f}")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


def main():
    X, y = make_moons(n_samples=600, noise=0.22, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    models = [
        (
            "Linear SVM",
            RecursivePartitionClassifier(
                base_estimator=SVC(kernel="linear", C=1.0, class_weight="balanced")
            ),
        ),
        (
            "RBF SVM",
            RecursivePartitionClassifier(
                base_estimator=SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
            ),
        ),
        (
            "Regularized QDA",
            RecursivePartitionClassifier(
                base_estimator=EqualPriorQDA(reg_param=0.05)
            ),
        ),
        (
            "Bagged QDA",
            BaggedRecursivePartitionClassifier(
                estimator=RecursivePartitionClassifier(
                    base_estimator=EqualPriorQDA(reg_param=0.05)
                ),
                n_estimators=60,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, (name, model) in zip(axes.ravel(), models):
        plot_model(ax, name, model, X_train, y_train, X_test, y_test)
    figure.suptitle("Recursive classifier-driven partitioning on two moons")
    plt.show()


if __name__ == "__main__":
    main()
