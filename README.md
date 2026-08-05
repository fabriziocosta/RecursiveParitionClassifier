# Recursive Partition Classifier

Scikit-learn-compatible binary and multiclass classifiers that recursively partition data
using predictions from a configurable base estimator, plus an optional
parallel bagging ensemble.

## Install

```bash
python -m pip install -e .
```

For the notebook and plotting example:

```bash
python -m pip install -e '.[demo]'
```

## Quick start

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

Swap in another classifier without changing the recursive partitioner:

```python
from recursive_partition import EqualPriorQDA

model = RecursivePartitionClassifier(
    base_estimator=EqualPriorQDA(reg_param=0.05)
)
```

`EqualPriorQDA` recalculates uniform priors from the classes present at each
recursive node, so it works for both binary and multiclass data.

For scikit-learn artificial neural networks, use the MLP adaptor. It exposes
the same constructor parameters as `MLPClassifier` and provides a stable
`sample_weight` fit argument across scikit-learn versions. It also balances
classes within each recursive node by default, preventing child MLPs from
collapsing to their local majority class:

```python
from sklearn.neural_network import MLPClassifier
from recursive_partition import MLPClassifierAdapter

model = RecursivePartitionClassifier(
    base_estimator=MLPClassifierAdapter(
        hidden_layer_sizes=(32, 16),
        max_iter=500,
        class_weight="balanced",
        random_state=42,
    )
)
```

`class_weight=None` disables balancing, while a class-weight dictionary can
provide custom local weights. `MLPClassifier` can also be supplied directly
when this recursive-node balancing is not needed. `MLPClassifierAdaptor` is
an equivalent spelling.

Gaussian-process classification is available through an adaptor with a
`ConstantKernel * Matern` default and the same local balancing behavior:

```python
from recursive_partition import GaussianProcessClassifierAdapter

model = RecursivePartitionClassifier(
    base_estimator=GaussianProcessClassifierAdapter(
        optimizer=None,
        class_weight="balanced",
        random_state=42,
    )
)
```

Because Gaussian-process fitting scales cubically with the number of samples,
it is best suited to smaller recursive nodes or datasets. Set
`class_weight=None` to disable adaptor resampling.

## Bagging

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

Each member is an independent job: it receives its own bootstrap sample,
random seed, estimator clone, and fit. The ensemble averages member
probabilities, so fitting is embarrassingly parallel through `n_jobs`. It does
not boost or sequentially reweight observations.

## Examples and API

- [2D dataset comparison notebook](notebooks/recursive_partition_datasets_demo.ipynb)
- [Executable plotting example](examples/plot_two_moons.py)
- [Architecture and full design specification](ARCHITECTURE.md)

The main classes are exported from `recursive_partition`:

```python
from recursive_partition import (
    RecursivePartitionClassifier,
    BaggedRecursivePartitionClassifier,
)
```

The classifiers support binary and multiclass targets, dense and compatible
CSR/CSC inputs, and expose `fit`, `predict`, `predict_proba`, `apply`,
`decision_path`, `get_depth`, and `get_n_leaves` where applicable. The package
also includes the self-contained `make_2d_dataset` and
`plot_probability_heatmap` helpers used by the notebook. See
`ARCHITECTURE.md` for stopping rules, probability modes, sample weights,
safeguards, and limitations.

## Development

```bash
python -m pip install -e '.[dev]'
pytest -q
```
