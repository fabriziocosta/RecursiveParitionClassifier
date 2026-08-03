You are to implement a production-quality, scikit-learn-compatible classifier for recursive top-down partitioning with modular binary base estimators, plus an optional parallel bagging ensemble.

The implementation must support plugging in classifiers such as:

```python
SVC(kernel="linear", class_weight="balanced")
SVC(kernel="rbf", class_weight="balanced")
SVC(kernel="poly", degree=2, class_weight="balanced")
QuadraticDiscriminantAnalysis(priors=[0.5, 0.5], reg_param=0.05)
LinearDiscriminantAnalysis(priors=[0.5, 0.5])
```

Do not hard-code logic specifically for SVM or QDA except where estimator capability detection is required.

## Core recursive classifier

Implement:

```python
RecursivePartitionClassifier
```

It must inherit from:

```python
sklearn.base.BaseEstimator
sklearn.base.ClassifierMixin
```

### Algorithm

For a binary classification problem, at every node:

1. Take the subset of training observations routed to that node.
2. Fit a fresh clone of `base_estimator` using:

   * the observations in that node;
   * their original ground-truth targets, not the parent predictions.
3. Predict the labels of the same node observations using the fitted estimator.
4. Define the two child partitions from those predicted labels:

   * negative child: observations predicted as `classes_[0]`;
   * positive child: observations predicted as `classes_[1]`.
5. Recursively repeat the procedure independently in both children.

The principal stopping rules are:

```text
- all observations at the node have the same true class;
- the node contains one observation.
```

Also include necessary feasibility and structural guards:

```text
- the base estimator cannot be fitted on the node;
- the estimator predicts only one class for every node observation;
- either child is empty;
- a child receives the entire parent subset;
- the estimator requires more observations per class than are available;
- numerical failure, singular covariance, or invalid predictions occur.
```

These guards must prevent infinite recursion without introducing CART-style impurity criteria. Do not use Gini impurity, entropy, information gain, or split optimization beyond fitting the supplied estimator.

### Estimator modularity

The constructor should be similar to:

```python
RecursivePartitionClassifier(
    base_estimator=None,
    probability_mode="leaf_frequency",
    probability_smoothing=1.0,
    on_fit_failure="leaf",
)
```

Use this default when `base_estimator=None`:

```python
SVC(kernel="linear", class_weight="balanced")
```

Use:

```python
sklearn.base.clone
```

to create an independent estimator at every internal node.

The implementation must work with estimators that:

* implement `fit` and `predict`;
* optionally implement `predict_proba`;
* optionally implement `decision_function`;
* may or may not accept `sample_weight`.

Detect capabilities safely rather than relying on estimator class names.

### Class imbalance

Do not modify arbitrary user-supplied estimators silently.

Document that imbalance treatment belongs to the base estimator configuration. Examples:

```python
SVC(class_weight="balanced")
QuadraticDiscriminantAnalysis(priors=[0.5, 0.5], reg_param=0.05)
```

For QDA, equal priors should be user-configurable, not automatically imposed by the recursive classifier.

### QDA feasibility

QDA with a full covariance matrix can fail in small or rank-deficient nodes.

Handle such failures gracefully by turning the node into a leaf when the estimator cannot be fitted. Do not crash the entire tree.

Do not hard-code a fixed minimum sample count globally. Prefer:

1. attempting the estimator fit;
2. catching well-defined numerical or validation failures;
3. applying optional configurable feasibility checks.

Expose an optional callable:

```python
node_fit_validator=None
```

with signature:

```python
node_fit_validator(estimator, X_node, y_node) -> bool
```

This allows users to add estimator-specific constraints.

## Prediction

Implement efficient batched traversal.

Do not route one sample at a time through repeated calls to `predict`.

Instead, recursively or iteratively maintain arrays of sample indices:

```python
stack = [(root, np.arange(n_samples))]
```

At each internal node:

1. call the node estimator’s `predict` once for the complete subset;
2. dispatch the corresponding index arrays to the two children.

Implement:

```python
fit
predict
predict_proba
decision_path
apply
get_depth
get_n_leaves
```

`apply(X)` should return terminal node IDs.

`decision_path(X)` may return either:

* a SciPy sparse node-indicator matrix; or
* a clearly documented list of node paths.

Prefer a CSR matrix compatible with scikit-learn conventions.

## Probabilities

Support at least these modes:

```python
probability_mode="leaf_frequency"
probability_mode="base_estimator"
```

### Leaf-frequency mode

At the terminal leaf, estimate:

[
p_k =
\frac{n_k+\alpha}
{n+K\alpha},
]

where `alpha=probability_smoothing`.

### Base-estimator mode

When possible, use the terminal routing estimator’s probability output or maintain the final internal-node probability associated with the selected branch.

Because this interpretation is less canonical, document it carefully.

Default to:

```python
probability_mode="leaf_frequency"
```

Ensure that:

```python
predict_proba(X).shape == (n_samples, 2)
predict_proba(X).sum(axis=1) == 1
```

up to numerical precision.

## Sample weights

`fit` should accept:

```python
fit(X, y, sample_weight=None)
```

If `sample_weight` is supplied:

* pass the node-specific subset of weights to estimators that support it;
* raise a clear error or follow a configurable policy for estimators that do not support it;
* do not assume every estimator accepts `sample_weight`.

Use scikit-learn utilities or signature inspection to detect support robustly.

## Parallel bagging ensemble

Implement:

```python
BaggedRecursivePartitionClassifier
```

It must also inherit from:

```python
BaseEstimator
ClassifierMixin
```

Suggested constructor:

```python
BaggedRecursivePartitionClassifier(
    estimator=None,
    n_estimators=30,
    max_samples=1.0,
    bootstrap=True,
    n_jobs=None,
    random_state=None,
    aggregation="mean_proba",
    oob_score=False,
    verbose=0,
)
```

Here, `estimator` should normally be a configured `RecursivePartitionClassifier`.

Example:

```python
tree = RecursivePartitionClassifier(
    base_estimator=SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
    )
)

ensemble = BaggedRecursivePartitionClassifier(
    estimator=tree,
    n_estimators=30,
    max_samples=1.0,
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
)
```

### Bagging procedure

For every ensemble member:

1. draw a bootstrap sample of the training data;
2. ensure both classes are represented;
3. clone the recursive estimator;
4. fit the clone on the bootstrap sample;
5. store the bootstrap indices and, when requested, out-of-bag indices.

Aggregate with an unweighted arithmetic mean:

[
\hat p(y=k\mid x)
=================

\frac{1}{B}
\sum_{b=1}^{B}
\hat p_b(y=k\mid x).
]

Do not implement boosting or sequential reweighting in this class.

### Parallelization

Use:

```python
joblib.Parallel
joblib.delayed
```

or a scikit-learn-supported parallel backend.

Parallelize independent ensemble-member fitting over `n_estimators`.

Requirements:

* deterministic results for a fixed `random_state`, regardless of `n_jobs`;
* generate one independent seed per estimator before launching workers;
* avoid sharing mutable random-state objects between workers;
* avoid copying unnecessarily large arrays where joblib memmapping can help;
* preserve estimator ordering;
* surface worker exceptions clearly.

Prediction may also be parallelized over estimators, but avoid excessive overhead for small datasets.

Use `sklearn.utils.check_random_state` or `numpy.random.SeedSequence` correctly.

## Scikit-learn compatibility

The estimators must satisfy standard scikit-learn conventions:

* constructor arguments are assigned directly without validation or mutation;
* no learned attributes are created in `__init__`;
* learned attributes end with `_`;
* `fit` returns `self`;
* use `check_X_y`, `check_array`, `check_is_fitted`, and `validate_data` where appropriate;
* set `classes_`, `n_features_in_`, and optionally `feature_names_in_`;
* support `get_params` and `set_params`, including nested parameters such as:

```python
base_estimator__C
estimator__base_estimator__gamma
```

* support use inside `Pipeline`;
* support `GridSearchCV`, `RandomizedSearchCV`, and `cross_val_score`;
* preserve arbitrary binary class labels such as strings or negative integers;
* reject multiclass input with a clear error;
* handle dense NumPy arrays and CSR/CSC sparse matrices when the supplied base estimator supports them;
* include `_more_tags` or `__sklearn_tags__` only if needed for the installed scikit-learn version.

Run:

```python
from sklearn.utils.estimator_checks import check_estimator
```

and document any justified limitations.

## Tree representation

Use an internal node data structure, for example:

```python
@dataclass
class _Node:
    node_id: int
    depth: int
    n_samples: int
    class_counts: np.ndarray
    predicted_class_index: int
    estimator: object | None
    negative_child: _Node | None
    positive_child: _Node | None
    is_leaf: bool
```

Avoid storing full training subsets inside every node. Store compact metadata and routing structure.

If training indices are retained for diagnostics, make this optional because it can substantially increase memory use.

## Robustness

Handle:

* arbitrary binary labels;
* one-class child nodes;
* singleton nodes;
* estimator fit failures;
* estimator predictions outside the known label set;
* zero-sized partitions;
* non-finite sample weights;
* reproducible bootstrapping;
* sparse input;
* singular QDA covariance matrices;
* extremely deep trees.

Avoid Python recursion limits where possible. Prefer iterative tree construction or document and protect against pathological depth.

Expose an optional safety parameter:

```python
max_depth=None
```

It should default to `None`, preserving the conceptual stopping rule, but permit users to bound pathological recursion.

Also expose:

```python
max_nodes=None
```

as an optional computational guard.

Do not silently impose a shallow default.

## Performance

Prediction must use batched routing.

For bagging:

* parallelize fitting;
* optionally parallelize prediction;
* avoid per-sample Python loops;
* minimize repeated validation;
* avoid repeated cloning inside prediction;
* use compact integer node IDs;
* make probability aggregation vectorized.

Provide brief complexity comments for:

* recursive-tree training;
* tree prediction;
* bagging training;
* bagging prediction.

## Required files

Produce a small package structure:

```text
recursive_partition/
    __init__.py
    tree.py
    ensemble.py
    _node.py
    validation.py
tests/
    test_tree.py
    test_ensemble.py
examples/
    plot_two_moons.py
pyproject.toml
README.md
```

## Tests

Use `pytest`.

Include tests for:

1. basic binary fit and prediction;
2. arbitrary string labels;
3. uniform-class stopping;
4. singleton stopping;
5. structural non-splitting guard;
6. QDA numerical failure becoming a leaf;
7. linear SVM plug-in;
8. RBF SVM plug-in;
9. QDA plug-in with equal priors and regularization;
10. probability normalization;
11. `apply`;
12. batched prediction matching a simple reference traversal;
13. reproducibility with fixed seeds;
14. identical bagging results for `n_jobs=1` and `n_jobs=2`;
15. nested parameter access;
16. compatibility with `Pipeline`;
17. compatibility with `GridSearchCV`;
18. bootstrap sample sizes;
19. both classes present in every accepted bootstrap sample;
20. OOB scoring, when implemented;
21. sparse input where supported;
22. sample-weight propagation;
23. clear failure when the base estimator lacks required methods.

## Example script

Create an executable two-moons example using:

```python
X, y = make_moons(
    n_samples=600,
    noise=0.22,
    random_state=42,
)
```

Use a stratified 70/30 train-test split with `random_state=42`.

Show at least these configurations:

```python
RecursivePartitionClassifier(
    base_estimator=SVC(
        kernel="linear",
        C=1.0,
        class_weight="balanced",
    )
)
```

```python
RecursivePartitionClassifier(
    base_estimator=SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
    )
)
```

```python
RecursivePartitionClassifier(
    base_estimator=QuadraticDiscriminantAnalysis(
        priors=[0.5, 0.5],
        reg_param=0.05,
    )
)
```

and the bagged form:

```python
BaggedRecursivePartitionClassifier(
    estimator=RecursivePartitionClassifier(
        base_estimator=QuadraticDiscriminantAnalysis(
            priors=[0.5, 0.5],
            reg_param=0.05,
        )
    ),
    n_estimators=60,
    n_jobs=-1,
    random_state=42,
)
```

Plot:

* an `RdBu_r` probability heatmap;
* a black 0.5 contour;
* training points with white marker edges;
* test accuracy in the title.

## Documentation

The README must explain the distinction between:

* recursive partitioning;
* ordinary decision trees;
* bagging;
* boosting.

State explicitly that the recursive split is generated by the fitted base classifier’s predictions and is not selected by Gini impurity or information gain.

Include examples showing how to swap SVM and QDA without changing the recursive-tree implementation.

## Deliverables

Return:

1. complete source code for every file;
2. tests;
3. example script;
4. installation instructions;
5. usage examples;
6. a brief design explanation;
7. known limitations;
8. benchmark results for the two-moons example;
9. confirmation that tests were run, including the exact commands and outputs.

Do not provide pseudocode only. Produce runnable, typed, documented Python code.

## Installation

From a checkout of this repository:

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e '.[dev]'
pytest -q
```

The public API is exported from `recursive_partition`:

```python
from recursive_partition import (
    BaggedRecursivePartitionClassifier,
    RecursivePartitionClassifier,
)
```

### Design notes

This implementation is a recursive partitioner, not an ordinary decision
tree. At each internal node it fits the configured base classifier on the
true labels in that node and uses that classifier's predictions as the split.
The split is not selected with Gini impurity, entropy, information gain, or a
threshold search. A fresh `clone(base_estimator)` is used at every node.

Bagging fits independent recursive partitioners on bootstrap samples and
averages their two-column probability outputs. Boosting is different: it fits
models sequentially while changing observation weights to focus on prior
errors. This package does not implement boosting or sequential reweighting.

Swap base estimators without changing the recursive implementation:

```python
linear = RecursivePartitionClassifier(
    base_estimator=SVC(kernel="linear", class_weight="balanced")
)
qda = RecursivePartitionClassifier(
    base_estimator=QuadraticDiscriminantAnalysis(
        priors=[0.5, 0.5], reg_param=0.05
    )
)
```

Imbalance handling remains the responsibility of the supplied estimator. The
recursive classifier does not silently alter user estimator parameters. For
example, use `class_weight="balanced"` for SVC or equal `priors` for QDA when
that is appropriate. With `probability_mode="base_estimator"`, the final
routing estimator's probability output is used when available; otherwise the
hard routing decision is used. `leaf_frequency` is the default and applies
the documented smoothed class frequency at the reached leaf.

### Complexity and limitations

If a node estimator costs `F(m, d)` to fit on `m` samples and `d` features,
training costs the sum of `F(m, d)` over internal nodes, plus routing costs
linear in the samples visited at each level. Prediction is batched by node,
so each internal node calls `predict` once for its complete routed subset.
Bagging multiplies training by the number of members and distributes those
independent fits with joblib; ensemble prediction averages one batched
prediction per member.

The implementation supports dense arrays and CSR/CSC matrices when the base
estimator supports them. It is intentionally binary-only. Estimator-specific
small-sample requirements are handled by turning failed nodes into leaves by
default; `on_fit_failure="raise"` is available for strict diagnostics, and
`node_fit_validator` can add optional feasibility checks. Extremely deep
trees can be bounded with `max_depth` or `max_nodes`. Estimators that do not
accept sample weights require `sample_weight_policy="ignore"` when weights
are supplied, or fail clearly with the default `"raise"` policy.

### Two-moons benchmark

Run the included executable example with:

```bash
python examples/plot_two_moons.py
```

It uses the requested 600-sample, noise-0.22 dataset and stratified 70/30
split, and reports the measured test accuracy in each plot title. Exact
accuracy depends on the installed scikit-learn version and estimator
implementation. With Python 3.14, scikit-learn 1.9.0, and the requested
`random_state=42` split, one run produced:

| Configuration | Test accuracy |
| --- | ---: |
| Linear SVM | 0.955556 |
| RBF SVM | 0.938889 |
| Regularized QDA | 0.955556 |
| 60-member bagged QDA | 0.966667 |

These are benchmark observations, not fixed API guarantees.

### Verification

The repository includes pytest coverage for tree construction, safeguards,
plug-in estimators, sparse input, sample weights, probability normalization,
pipelines, parameter search, bagging reproducibility, bootstrap class
coverage, and OOB scoring. The verification command is:

```bash
pytest -q
```

The installed scikit-learn 1.9.0 `check_estimator(RecursivePartitionClassifier())`
suite includes a one-label training case and therefore raises the intentional
binary-target error. This package documents that limitation: multiclass and
one-class training inputs are rejected; normal binary estimator checks and the
broader pipeline/search behaviors are covered by the included tests.
