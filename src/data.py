"""Load the credit card fraud data, split it, and build the 55 engineered features.

Protocol: stratified 60 / 20 / 20 train / validation / test split with a fixed seed. Everything that learns from
data (feature quantiles, time range, scaler, SMOTE, models, decision thresholds) sees training or validation rows
only; the test rows are scored once, at the end.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

RAW_COLUMNS = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]


def load_raw(verify: bool = True) -> pd.DataFrame:
    """The 284,807 transactions with columns Time, V1-V28, Amount, Class (1 = fraud).

    Uses creditcard.csv from Kaggle if it is in the repo root; otherwise downloads the same data from OpenML
    (dataset 1597, ~150 MB) and caches it as data/creditcard.parquet.
    """
    if config.KAGGLE_CSV.exists():
        df = pd.read_csv(config.KAGGLE_CSV)
    elif config.CACHE_PARQUET.exists():
        df = pd.read_parquet(config.CACHE_PARQUET)
    else:
        df = download_openml()
        config.DATA_DIR.mkdir(exist_ok=True)
        df.to_parquet(config.CACHE_PARQUET)
    df = df[[*RAW_COLUMNS, "Class"]].copy()
    df["Class"] = df["Class"].astype(str).str.strip("'\"").astype(float).astype(int)
    df[RAW_COLUMNS] = df[RAW_COLUMNS].astype(float)
    if verify:
        check_dataset(df)
    return df


def read_arff(lines) -> pd.DataFrame:
    """Minimal reader for OpenML's numeric ARFF files: @attribute names, then comma-separated @data rows."""
    import io

    names, data_start = [], None
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if low.startswith("@attribute"):
            names.append(line.split()[1].strip("'\""))
        elif low.startswith("@data"):
            data_start = i + 1
            break
    if data_start is None:
        raise ValueError("no @data section in the ARFF file")
    body = "".join(l for l in lines[data_start:] if l.strip() and not l.startswith("%"))
    return pd.read_csv(io.StringIO(body), header=None, names=names, quotechar="'")


def download_openml() -> pd.DataFrame:
    import urllib.request

    print(f"Downloading {config.OPENML_ARFF_URL} ...", flush=True)
    with urllib.request.urlopen(config.OPENML_ARFF_URL, timeout=300) as resp:
        text = resp.read().decode("utf-8")
    return read_arff(text.splitlines(keepends=True))


def check_dataset(df: pd.DataFrame) -> None:
    n, fraud = len(df), int(df["Class"].sum())
    if (n, fraud) != (config.EXPECTED_ROWS, config.EXPECTED_FRAUD):
        raise ValueError(f"Unexpected dataset: {n:,} rows and {fraud} frauds; expected "
                         f"{config.EXPECTED_ROWS:,} rows and {config.EXPECTED_FRAUD} frauds.")


def split(df: pd.DataFrame, seed: int = config.SEED):
    """Stratified train / validation / test split (60 / 20 / 20 by default)."""
    rest, test = train_test_split(df, test_size=config.TEST_SIZE, stratify=df["Class"], random_state=seed)
    val_share = config.VAL_SIZE / (1 - config.TEST_SIZE)
    train, val = train_test_split(rest, test_size=val_share, stratify=rest["Class"], random_state=seed)
    return train, val, test


class FeatureBuilder:
    """The notebook's feature engineering (30 raw -> 55 features), with its two data-dependent steps (amount
    quintile edges and the time range used to derive the hour) learned from the training rows only."""

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        amount_log = np.log1p(df["Amount"])
        edges = np.unique(np.quantile(amount_log, [0.2, 0.4, 0.6, 0.8]))
        self.amount_edges_ = edges
        self.time_min_, self.time_max_ = float(df["Time"].min()), float(df["Time"].max())
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[[c for c in df.columns if c.startswith("V")]].copy()
        amount_log = np.log1p(df["Amount"])

        # 1. Amount
        out["Amount_sq"] = amount_log ** 2
        out["Amount_isZero"] = (amount_log.abs() < 1e-6).astype(int)
        # Quintile bins; values beyond the training range fall in the first or last bin.
        out["Amount_bin"] = np.searchsorted(self.amount_edges_, amount_log, side="left")  # (a, b] bins, like pd.qcut

        # 2. Time: the ~48 hours of data mapped onto hours of the day
        span = (self.time_max_ - self.time_min_) or 1.0
        time_norm = ((df["Time"] - self.time_min_) / span).clip(0, 1)
        hour = (time_norm * 47).astype(int) % 24
        out["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out["Is_night"] = hour.between(0, 5).astype(int)
        out["Is_peak"] = hour.between(9, 17).astype(int)

        # 3. V-feature interactions
        out["V1_V2_ratio"] = df["V1"] / (df["V2"].abs() + 1e-6)
        out["V3_V4_ratio"] = df["V3"] / (df["V4"].abs() + 1e-6)
        out["V14_V17_prod"] = df["V14"] * df["V17"]
        out["V2_V4_prod"] = df["V2"] * df["V4"]
        out["V11_V3_prod"] = df["V11"] * df["V3"]
        for v in ["V1", "V2", "V3", "V4", "V10", "V11", "V12", "V14", "V17"]:
            out[f"{v}_abs"] = df[v].abs()

        # 4. Aggregate fraud signal
        out["fraud_neg_sum"] = df[["V3", "V14", "V17", "V12", "V10"]].sum(axis=1)
        out["fraud_pos_sum"] = df[["V2", "V4", "V11"]].sum(axis=1)
        out["fraud_signal"] = out["fraud_pos_sum"] - out["fraud_neg_sum"]

        # 5. Amount x V interactions
        out["Amount_x_V14"] = amount_log * df["V14"]
        out["Amount_x_V17"] = amount_log * df["V17"]
        out["Amount_x_V4"] = amount_log * df["V4"]
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
