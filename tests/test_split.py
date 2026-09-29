import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import FeatureBuilder, check_dataset, split


def test_split_is_60_20_20_disjoint_and_complete(toy):
    train, val, test = split(toy)
    assert (len(train), len(val), len(test)) == (1200, 400, 400)
    idx = [set(train.index), set(val.index), set(test.index)]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])
    assert set().union(*idx) == set(toy.index)


def test_split_is_stratified(toy):
    train, val, test = split(toy)
    assert (train.Class.sum(), val.Class.sum(), test.Class.sum()) == (24, 8, 8)


def test_split_is_reproducible(toy):
    a, b = split(toy), split(toy)
    for x, y in zip(a, b):
        assert list(x.index) == list(y.index)
    assert list(split(toy, seed=1)[2].index) != list(a[2].index)


def test_check_dataset_accepts_only_the_expected_counts():
    ok = pd.DataFrame({"Class": [1] * config.EXPECTED_FRAUD + [0] * (config.EXPECTED_ROWS - config.EXPECTED_FRAUD)})
    check_dataset(ok)
    with pytest.raises(ValueError, match="Unexpected dataset"):
        check_dataset(ok.iloc[:-1])


def test_features_are_the_notebooks_55(toy):
    X = FeatureBuilder().fit_transform(toy)
    assert X.shape[1] == 55
    assert "Class" not in X and "Amount" not in X and "Time" not in X
    assert not X.isna().any().any()
    np.testing.assert_allclose(X["fraud_signal"], toy[["V2", "V4", "V11"]].sum(axis=1) -
                               toy[["V3", "V14", "V17", "V12", "V10"]].sum(axis=1))


def test_feature_statistics_come_from_training_rows_only(toy):
    train, val, test = split(toy)
    fb = FeatureBuilder().fit(train)
    edges = np.quantile(np.log1p(train["Amount"]), [0.2, 0.4, 0.6, 0.8])
    np.testing.assert_allclose(fb.amount_edges_, np.unique(edges))
    assert fb.time_min_ == train["Time"].min() and fb.time_max_ == train["Time"].max()
    # Transforming test rows doesn't change what was learned.
    fb.transform(test)
    np.testing.assert_allclose(fb.amount_edges_, np.unique(edges))


def test_amount_bins_match_pandas_qcut_on_the_fitting_data(toy):
    X = FeatureBuilder().fit_transform(toy)
    expected = pd.qcut(np.log1p(toy["Amount"]), q=5, labels=False, duplicates="drop")
    assert (X["Amount_bin"].to_numpy() == expected.to_numpy()).all()


def test_read_arff_keeps_every_column_including_time():
    from src.data import read_arff

    arff = """% comment
@relation creditcard
@attribute 'Time' numeric
@attribute 'V1' numeric
@attribute 'Class' {'0','1'}
@data
0,-1.35,'0'
% another comment
1,1.19,'1'
"""
    df = read_arff(arff.splitlines(keepends=True))
    assert list(df.columns) == ["Time", "V1", "Class"]
    assert df["Time"].tolist() == [0, 1] and df["Class"].tolist() == [0, 1]
