import numpy as np
from sklearn.svm import SVC

from recursive_partition import BaggedRecursivePartitionClassifier, RecursivePartitionClassifier


def data():
    rng = np.random.RandomState(12)
    X = rng.normal(size=(100, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y


def make_model(n_jobs):
    return BaggedRecursivePartitionClassifier(
        estimator=RecursivePartitionClassifier(base_estimator=SVC(kernel="linear", probability=True)),
        n_estimators=6,
        random_state=42,
        n_jobs=n_jobs,
        oob_score=True,
    )


def test_reproducible_across_parallelism_and_bootstrap_shape():
    X, y = data()
    serial = make_model(1).fit(X, y)
    parallel = make_model(2).fit(X, y)
    np.testing.assert_allclose(serial.predict_proba(X), parallel.predict_proba(X))
    assert all(len(indices) == len(X) for indices in serial.estimators_samples_)
    assert all(len(np.unique(y[indices])) == 2 for indices in serial.estimators_samples_)
    np.testing.assert_array_equal(serial.estimator_seeds_, parallel.estimator_seeds_)


def test_oob_scoring_and_nested_parameters():
    X, y = data()
    model = make_model(1)
    assert model.get_params()["estimator__base_estimator__kernel"] == "linear"
    model.fit(X, y)
    assert model.oob_decision_function_.shape == (len(y), 2)
    assert 0 <= model.oob_score_ <= 1
    assert model.predict(X).shape == y.shape


def test_sample_weight_propagates_through_ensemble():
    X, y = data()
    model = BaggedRecursivePartitionClassifier(
        estimator=RecursivePartitionClassifier(base_estimator=SVC(kernel="linear")),
        n_estimators=2,
        random_state=1,
    )
    model.fit(X, y, sample_weight=np.ones(len(y)))
    assert model.predict(X).shape == y.shape


def test_each_member_gets_its_independent_seed():
    X, y = data()
    model = BaggedRecursivePartitionClassifier(
        estimator=RecursivePartitionClassifier(
            base_estimator=SVC(kernel="rbf", probability=True, random_state=None)
        ),
        n_estimators=4,
        random_state=42,
        n_jobs=1,
    ).fit(X, y)
    member_seeds = np.asarray([estimator.base_estimator_.random_state for estimator in model.estimators_])
    np.testing.assert_array_equal(member_seeds, model.estimator_seeds_)
    assert len(np.unique(member_seeds)) == len(member_seeds)
