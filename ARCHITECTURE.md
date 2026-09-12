# Implemented architecture

The package provides scikit-learn-compatible classifiers that build a tree from
the predictions of a configurable classifier. It also provides a parallel
bagging wrapper, a QDA variant for locally uniform priors, two classifier
adaptors with local class balancing, synthetic 2-D datasets, and plotting
helpers.

## Package layout

```text
recursive_partition/
    __init__.py
    tree.py                 RecursivePartitionClassifier
    ensemble.py             BaggedRecursivePartitionClassifier
    _node.py                Internal node dataclass
    validation.py           Target, weight, and capability checks
    _balancing.py           Local weighting and resampling helpers
    qda.py                  EqualPriorQDA
    neural_network.py       MLPClassifierAdapter / Adaptor
    gaussian_process.py     GaussianProcessClassifierAdapter / Adaptor
    datasets.py             make_2d_dataset
    plotting.py             plot_probability_heatmap
tests/
examples/plot_two_moons.py
```

The public objects are re-exported from `recursive_partition.__init__`.

## RecursivePartitionClassifier

`RecursivePartitionClassifier` inherits from `ClassifierMixin` and
`BaseEstimator`. Its constructor is:

```python
RecursivePartitionClassifier(
    base_estimator=None,
    probability_mode="leaf_frequency",
    probability_smoothing=1.0,
    on_fit_failure="leaf",
    node_fit_validator=None,
    max_depth=None,
    max_nodes=None,
    sample_weight_policy="raise",
)
```

Constructor values are assigned unchanged, which preserves normal
scikit-learn cloning and nested parameter access. No fitted state is created in
`__init__`.

When `base_estimator` is `None`, fitting uses a fresh
`SVC(kernel="linear", class_weight="balanced")` template. A supplied estimator
must expose callable `fit` and `predict` methods. The implementation does not
special-case SVMs, logistic regression, QDA, or other classifier names.

### Training algorithm

`fit(X, y, sample_weight=None)` first validates dense or CSR/CSC input,
requires a binary or multiclass target with at least two classes, and records
`classes_`, `n_classes_`, and `n_features_in_`. It then builds the tree
iteratively with a stack, avoiding Python call-stack recursion.

For each non-terminal node:

1. The node’s row indices select `X_node` and the original labels `y_node`.
2. A fresh `clone(base_estimator)` is made.
3. The optional `node_fit_validator(estimator, X_node, y_node)` is called. A
   false result leaves the node terminal.
4. The estimator is fitted on the node’s original labels. It is not fitted on
   predictions made by its parent.
5. The fitted estimator predicts the complete node subset in one call.
6. Predictions are mapped to the global `classes_` positions and become the
   routing labels for the children.

The split is therefore classifier-driven. There is no CART impurity calculation,
threshold search, Gini criterion, entropy, information gain, boosting, or
sequential error reweighting in this class.

The implementation creates one child for every class in the union of the
classes predicted at the node and the true classes present in the node. The
predicted classes always have non-empty routed subsets. A true local class that
the estimator never predicted receives an empty fallback child; that child is
made terminal and inherits the parent’s class-count vector. This preserves a
complete class-keyed routing structure and gives traversal somewhere to go if a
later prediction uses that class label.

### Terminal-node rules and safeguards

A node is left terminal when any of these conditions applies:

- it contains one or zero observations;
- all observations have the same true class;
- `max_depth` has been reached;
- the estimator predicts fewer than two distinct known classes;
- `node_fit_validator` returns `False`;
- fitting or in-node prediction raises an exception, when
  `on_fit_failure="leaf"`;
- expanding the node would exceed `max_nodes`.

Unknown labels, an incorrectly sized prediction, missing children during
traversal, and other estimator errors are treated as failures during fitting or
prediction. With the default `on_fit_failure="leaf"`, fitting errors turn the
current node into a leaf. With `on_fit_failure="raise"`, the fitting error is
re-raised as a `RuntimeError` containing the node ID and original exception.

`max_depth=None` and `max_nodes=None` leave these computational guards
unbounded. `max_depth` is measured with the root at depth zero. `max_nodes`
limits the number of node objects created; the implementation declines an
expansion when the required children would exceed the limit.

The implementation does not impose a global minimum sample count. Estimator-
specific feasibility failures, such as singular or underspecified QDA nodes,
are discovered by attempting the fit. A callable `node_fit_validator` is
available for users who need an earlier estimator-specific check.

### Internal representation

Each `_Node` is a dataclass containing:

```text
node_id              Integer ID used by apply() and decision_path()
depth                Root-to-node depth
n_samples            Number of training rows routed to the node
class_counts         Unweighted or sample-weighted counts for global classes
predicted_class_index Majority class position used by predict() at a leaf
estimator             Fitted node estimator, or None for a leaf
children              Mapping from global class index to child node
negative_child        Binary-class convenience reference to child 0
positive_child        Binary-class convenience reference to child 1
is_leaf               Whether the node is terminal
```

Training subsets and row indices are not retained in the nodes after fitting.
The fitted estimators and compact class-count metadata are retained.

## Prediction and tree inspection

`predict(X)` performs batched traversal. It maintains a stack of `(node,
sample_indices)` pairs, calls each internal estimator once for all samples
routed to that node, and dispatches the resulting index arrays to child nodes.
It returns the majority true class recorded at each reached leaf, using the
original label values in `classes_`.

The same traversal machinery powers:

- `apply(X)`, which returns the terminal `node_id` for each sample;
- `decision_path(X)`, which returns a CSR matrix with one row per sample and
  one column per fitted node. Entries are 1 for nodes visited by that sample;
- `get_depth()`, which returns the maximum fitted depth;
- `get_n_leaves()`, which returns the number of terminal nodes.

Fitted-state and feature-count checks are applied at prediction time. Dense
arrays and CSR/CSC matrices are accepted when the supplied base estimator also
supports the corresponding representation.

## Probabilities

The default `probability_mode="leaf_frequency"` uses the class counts stored at
the reached leaf:

```text
(class_count + probability_smoothing)
-------------------------------------
(total_count + n_classes * probability_smoothing)
```

The default smoothing value is `1.0`. A value of zero is accepted. The result
is normalized and has one column for every global class, including classes
that are absent from a particular leaf.

With `probability_mode="base_estimator"`, traversal still ends at a leaf, but
the probability vector recorded for each sample is taken from the deepest
internal estimator visited for that sample. Thus, a deeper node overwrites the
probability produced by its parent. If that estimator exposes `predict_proba`,
its local class columns are mapped to the global class order. If it does not,
the hard routing prediction is represented as a one-hot vector. If the
probability call fails or returns unusable data, the hard routing result is
used. Samples that never pass through an internal estimator (for example, a
one-node tree) fall back to the leaf-frequency result.

Both modes clip negative values, normalize rows, and fall back to a uniform
distribution if a row cannot be normalized. `predict_proba(X)` therefore
returns an `(n_samples, n_classes_)` array whose rows sum to one up to floating-
point precision.

## Sample weights

`RecursivePartitionClassifier.fit` accepts a one-dimensional, finite,
non-negative `sample_weight` array with at least one positive value. Node class
counts are weighted when weights are supplied.

Before each node fit, the estimator’s `fit` signature is inspected. Weights are
passed as `sample_weight=...` when the estimator supports that argument,
including estimators that advertise support through
`_fit_accepts_sample_weight()`. If the estimator does not support weights:

- `sample_weight_policy="raise"` (the default) raises a clear `TypeError`;
- `sample_weight_policy="ignore"` fits without passing the weights.

Weights are sliced to the rows belonging to each node. The recursive classifier
does not silently alter arbitrary estimator parameters or class priors.

## BaggedRecursivePartitionClassifier

`BaggedRecursivePartitionClassifier` is also a `ClassifierMixin`/`BaseEstimator`
and implements independent bootstrap-style fitting:

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
    max_depth=None,
)
```

The default member is `RecursivePartitionClassifier()`. A supplied member must
implement both `fit` and `predict_proba`; the latter is required because the
ensemble aggregates probabilities. The fitted template is cloned into
`estimator_` for inspection, while the independent fitted members are stored in
`estimators_`.

### Sampling and fitting

`max_samples` may be an integer between 1 and `n_samples`, or a float in
`(0, 1]`. A float is converted with `ceil(max_samples * n_samples)`.
`max_depth` is an optional non-negative integer applied to each member. When a
custom estimator is supplied, it must expose a `max_depth` parameter whenever
`max_depth` is not `None`.

Every member’s sample is constructed so that it contains at least one training
row from every global class. With `bootstrap=True`, the remaining rows are
drawn with replacement. With `bootstrap=False`, the remaining rows are drawn
without replacement. The selected row IDs are retained in
`estimators_samples_`.

Before workers are launched, one integer seed is generated for each member.
The seeds are stored in `estimator_seeds_`, and the same seed is assigned to
all nested `random_state` parameters in that member clone. This makes fitting
deterministic for a fixed `random_state` and preserves member ordering across
different `n_jobs` settings.

Members are fitted independently using `joblib.Parallel` and `delayed` with a
process-based preference. A worker exception is surfaced by joblib. Supplied
sample weights are sliced by the member’s sampled indices and passed to the
member estimator; the weights do not affect how bootstrap indices are drawn.

The only supported aggregation value is `aggregation="mean_proba"`. Prediction
gets one probability matrix from each member, takes the unweighted arithmetic
mean, clips negative values, and normalizes the result. `predict` chooses the
highest-probability global class. Prediction is parallelized when there are
more than two members; smaller ensembles are evaluated serially to avoid
unnecessary process overhead.

### Out-of-bag results

`oob_indices_` is recorded for every member whether or not OOB scoring is
requested. When `oob_score=True`, the ensemble predicts each sample only with
members for which that sample is out of bag and stores:

- `oob_decision_function_`: mean OOB probabilities, with `NaN` rows when a
  sample has no OOB predictions;
- `oob_counts_`: the number of contributing members per sample;
- `oob_score_`: accuracy over samples with at least one OOB prediction.

## Estimator adaptors and helpers

### EqualPriorQDA

`EqualPriorQDA` subclasses scikit-learn’s
`QuadraticDiscriminantAnalysis`. Immediately before each fit it replaces
`priors` with a uniform vector whose length is the number of classes present in
that local node. This makes equal priors usable when recursive multiclass nodes
contain different class subsets. It rejects one-class local fits. It does not
accept `sample_weight` in its fit signature, so recursive weighted fitting
requires `sample_weight_policy="ignore"`.

Ordinary QDA, SVC, logistic regression, discriminant analysis, and other
estimators can be supplied directly. The recursive classifier leaves their
imbalance configuration unchanged.

### MLPClassifierAdapter

`MLPClassifierAdapter` subclasses `sklearn.neural_network.MLPClassifier` and
adds a `class_weight` parameter, defaulting to `"balanced"`. It computes local
class/sample weights at every recursive fit. If the installed scikit-learn MLP
supports `sample_weight`, those weights are passed through; otherwise the
adaptor applies deterministic weighted resampling. `class_weight=None`
disables adaptor-level balancing. `MLPClassifierAdaptor` is an alias.

The adaptor keeps the MLP constructor parameters exposed for cloning, pipelines,
grid searches, and nested parameter access.

### GaussianProcessClassifierAdapter

`GaussianProcessClassifierAdapter` subclasses
`sklearn.gaussian_process.GaussianProcessClassifier`, adds the same local
`class_weight` behavior, and defaults to a
`ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)` kernel. Since the
scikit-learn Gaussian-process classifier does not expose `sample_weight`, local
weights are applied through deterministic weighted resampling. The alias
`GaussianProcessClassifierAdaptor` is also exported.

### Dataset and plotting utilities

`make_2d_dataset` supplies the package’s synthetic/demo datasets, including
moons, blobs, XOR, spirals, checkerboard, classification data, anisotropic
blobs, and an equal-isotropic-Gaussian fallback. `plot_probability_heatmap`
accepts a fitted classifier and 2-D data. Binary plots use an `RdBu_r`
probability surface, a black 0.5 contour, and a thinner configurable confidence
contour. Multiclass plots use probability-weighted `tab10` colors and entropy-
based confidence fading. Matplotlib is imported lazily.

The executable `examples/plot_two_moons.py` builds a 600-sample, noise-0.22
two-moons dataset, performs a stratified 70/30 split with `random_state=42`,
and plots linear SVM, RBF SVM, regularized equal-prior QDA, and 60-member
bagged QDA configurations with test accuracy in each title.

## Scikit-learn behavior

Both classifiers use `check_X_y`, `check_array`, and `check_is_fitted` where
appropriate, expose learned attributes with trailing underscores, return
`self` from `fit`, and support `get_params`/`set_params` through normal
scikit-learn introspection. Nested parameters such as the following work:

```python
tree.get_params()["base_estimator__C"]
ensemble.get_params()["estimator__base_estimator__gamma"]
```

They can be used in `Pipeline`, `GridSearchCV`, and related model-selection
APIs. Targets may use arbitrary labels such as strings or negative integers;
all internal routing uses integer positions while public predictions use the
original labels.

## Complexity and limitations

If fitting a node estimator costs `F(m, d)` for `m` rows and `d` features,
training costs the sum of `F(m, d)` over all successfully expanded nodes plus
the cost of routing each training subset. Prediction is batched by node, so an
internal node makes one prediction call for its complete routed subset. Bagging
multiplies member training by `n_estimators` and averages one batched
probability traversal per member.

Important current limitations are:

- The target must be binary or multiclass with at least two global classes;
  one-label training data is rejected.
- A base estimator must support `fit` and `predict`; a bagging member must
  additionally support `predict_proba`.
- Input support ultimately depends on the supplied estimator’s dense/sparse
  capabilities.
- `base_estimator` probability mode is a routing diagnostic interpretation,
  not a calibrated probability model. It uses the deepest internal estimator’s
  output and falls back to hard routing when necessary.
- QDA and other estimators with local sample-size or numerical requirements may
  stop early. The default is to turn failed nodes into leaves.
- `EqualPriorQDA` does not natively support sample weights.
- Ensemble aggregation is currently limited to arithmetic mean probabilities;
  voting, boosting, and sequential reweighting are not implemented.
- There is no explicit global minimum node size. Use `node_fit_validator`,
  `max_depth`, or `max_nodes` when a workload needs stronger bounds.
- The package does not claim full success under the complete generic
  `check_estimator` suite: one-label training is intentionally rejected by the
  target validator. The included tests cover the supported binary/multiclass,
  pipeline, search, sparse, weighting, and ensemble behaviors.

## Installation and use

From a checkout:

```bash
python -m pip install -e .
python -m pip install -e '.[dev]'  # development/test dependencies
pytest -q
```

Basic use:

```python
from sklearn.svm import SVC
from recursive_partition import RecursivePartitionClassifier

model = RecursivePartitionClassifier(
    base_estimator=SVC(kernel="rbf", class_weight="balanced")
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

The recursive implementation can be reused with another classifier:

```python
from recursive_partition import EqualPriorQDA

model = RecursivePartitionClassifier(
    base_estimator=EqualPriorQDA(reg_param=0.05),
    on_fit_failure="leaf",
)
```

For bagging:

```python
from recursive_partition import BaggedRecursivePartitionClassifier

ensemble = BaggedRecursivePartitionClassifier(
    estimator=model,
    n_estimators=30,
    n_jobs=-1,
    random_state=42,
)
ensemble.fit(X_train, y_train)
```

## Verification performed

The repository test suite was run in the project virtual environment with:

```bash
./.venv/bin/pytest -q
```

At the time this document was updated, the result was:

```text
25 passed, 55 warnings in 2.67s
```

The warnings are scikit-learn 1.9.0 deprecation warnings for the test use of
`SVC(probability=True)`. The test suite covers tree fitting and stopping,
arbitrary labels, estimator plug-ins, QDA failure handling, probabilities,
batched traversal, sparse input, sample weights, pipelines, grid search,
bootstrap coverage, reproducibility across `n_jobs`, nested random seeds, OOB
scoring, multiclass bagging, plotting, and invalid estimator errors.
