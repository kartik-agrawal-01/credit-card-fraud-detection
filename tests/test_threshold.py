import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.train import best_f1_threshold, choose_threshold, prepare
from src.data import split


def test_best_threshold_separates_perfectly_separable_scores():
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    thr, f1 = best_f1_threshold(y, scores)
    assert thr == pytest.approx(0.8) and f1 == pytest.approx(1.0, abs=1e-6)


def test_best_threshold_maximises_f1():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    scores = y * 0.3 + rng.uniform(size=300)
    thr, f1 = best_f1_threshold(y, scores)
    assert f1_score(y, scores >= thr) == pytest.approx(f1, abs=1e-6)
    for t in np.unique(scores):
        assert f1_score(y, scores >= t) <= f1 + 1e-6


def test_autoencoder_threshold_comes_from_the_training_error_grid():
    class AE:
        train_legit_mse_ = np.linspace(0, 1, 1001)
    y_val = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.5, 0.97, 0.99])
    thr, f1 = choose_threshold("autoencoder", AE(), y_val, scores)
    assert 0.5 < thr <= 0.97 and f1 == 1.0
    assert thr in [np.percentile(AE.train_legit_mse_, q) for q in np.arange(80, 100, 0.5)]


def test_smote_only_touches_the_training_rows(toy):
    train, val, test = split(toy)
    p = prepare(train, val, test)
    assert len(p.X_val) == len(val) and len(p.X_test) == len(test)
    assert p.y_val.sum() == val.Class.sum() and p.y_test.sum() == test.Class.sum()
    legit = (p.y_train == 0).sum()
    assert (p.y_train_sm == 0).sum() == legit
    assert p.y_train_sm.sum() == int(0.1 * legit)          # fraud upsampled to 10% of legitimate
    np.testing.assert_allclose(p.X_train_sm[: len(p.X_train)], p.X_train)  # originals kept, synthetic appended


def test_scaler_is_fitted_on_training_rows(toy):
    train, val, test = split(toy)
    p = prepare(train, val, test)
    np.testing.assert_allclose(p.X_train.mean(axis=0), 0, atol=1e-8)
    assert np.abs(p.X_test.mean(axis=0)).max() > 1e-3   # test rows are transformed, not re-fitted
