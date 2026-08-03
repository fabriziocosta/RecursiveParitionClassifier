import numpy as np
import pytest
from scipy import sparse
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from recursive_partition import RecursivePartitionClassifier


def data():
    rng = np.random.RandomState(4)
    X = rng.normal(size=(80, 3))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    return X, y


def test_basic_fit_prediction_and_probabilities():
    X, y = data()
    model = RecursivePartitionClassifier(base_estimator=SVC(kernel="linear", probability=True))
    model.fit(X, y)
    assert model.predict(X).shape == y.shape
    probabilities = model.predict_proba(X)
    assert probabilities.shape == (len(y), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    assert model.get_n_leaves() >= 1


def test_string_labels_and_apply_decision_path():
    X, y = data()
    labels = np.where(y == 0, "negative", "positive")
    model = RecursivePartitionClassifier(base_estimator=LogisticRegression(max_iter=500)).fit(X, labels)
    assert set(model.predict(X)) <= {"negative", "positive"}
    leaves = model.apply(X)
    path = model.decision_path(X)
    assert path.shape == (len(X), model.n_nodes_)
    assert np.all(path.sum(axis=1).A1 >= 1)
    assert np.all(np.asarray(path[np.arange(len(X)), leaves]).ravel() == 1)


def test_uniform_and_singleton_stopping():
    X, y = data()
    uniform = RecursivePartitionClassifier().fit(X, np.where(np.arange(len(y)) % 2, "a", "b"))
    assert uniform.get_n_leaves() >= 1
    singleton = RecursivePartitionClassifier().fit(X[:2], np.array([0, 1]))
    assert all(node.n_samples == 1 for node in singleton.nodes_ if node.is_leaf)


def test_structural_non_splitting_guard():
    X, y = data()
    model = RecursivePartitionClassifier(base_estimator=DummyClassifier(strategy="most_frequent")).fit(X, y)
    assert model.tree_.is_leaf
    assert model.n_nodes_ == 1


def test_qda_failure_becomes_leaf():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    model = RecursivePartitionClassifier(base_estimator=QuadraticDiscriminantAnalysis()).fit(X, y)
    assert model.n_nodes_ >= 1
    assert model.predict_proba(X).shape == (4, 2)


@pytest.mark.parametrize("estimator", [
    SVC(kernel="linear", class_weight="balanced"),
    SVC(kernel="rbf", class_weight="balanced"),
    QuadraticDiscriminantAnalysis(priors=[0.5, 0.5], reg_param=0.05),
])
def test_estimator_plugins(estimator):
    X, y = data()
    model = RecursivePartitionClassifier(base_estimator=estimator).fit(X, y)
    assert model.predict(X).shape == y.shape


def test_batched_traversal_matches_predict():
    X, y = data()
    model = RecursivePartitionClassifier(base_estimator=LogisticRegression(max_iter=500)).fit(X, y)
    batched = model.predict(X)
    reference = np.array([model.classes_[model._traverse(X[i : i + 1])[0][0].predicted_class_index] for i in range(len(X))])
    np.testing.assert_array_equal(batched, reference)


def test_sparse_input_and_sample_weight():
    X, y = data()
    model = RecursivePartitionClassifier(base_estimator=LogisticRegression(max_iter=500)).fit(
        sparse.csr_matrix(X), y, sample_weight=np.ones(len(y))
    )
    assert model.predict(sparse.csr_matrix(X)).shape == y.shape


def test_nested_params_pipeline_and_grid_search():
    X, y = data()
    model = RecursivePartitionClassifier(base_estimator=SVC(kernel="linear", C=1.0))
    assert model.get_params()["base_estimator__C"] == 1.0
    pipeline = Pipeline([("scale", StandardScaler()), ("model", model)])
    pipeline.fit(X, y)
    search = GridSearchCV(
        RecursivePartitionClassifier(base_estimator=SVC(kernel="linear")),
        {"base_estimator__C": [0.5, 1.0]},
        cv=2,
    ).fit(X, y)
    assert search.best_estimator_.classes_.shape == (2,)


def test_clear_failure_for_invalid_base_estimator():
    with pytest.raises(TypeError, match="fit and predict"):
        RecursivePartitionClassifier(base_estimator=object()).fit(*data())
