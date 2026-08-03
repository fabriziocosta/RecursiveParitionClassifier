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

- [2D dataset comparison notebook](notebooks/two_moons_demo.ipynb)
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
