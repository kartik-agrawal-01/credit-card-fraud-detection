"""Fit the six models from the notebook and choose each one's decision threshold on the validation set.

Hyperparameters are the notebook's. What changed is the protocol (see data.py): SMOTE is applied to the training
rows only, and thresholds are tuned on validation instead of on the test set.

Usage:
    python -m src.train                  # fit everything and print validation results
    python -m src.train --models xgboost random_forest
"""
import argparse
import os
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler

from . import config
from .data import FeatureBuilder, load_raw, split

MODELS = ["xgboost", "random_forest", "ann", "autoencoder", "isolation_forest", "lof"]
DISPLAY = {
    "xgboost": "XGBoost (SMOTE)",
    "random_forest": "Random Forest (SMOTE)",
    "ann": "ANN classifier (class weights)",
    "autoencoder": "Autoencoder (reconstruction error)",
    "isolation_forest": "Isolation Forest",
    "lof": "Local Outlier Factor",
}
DL_MODELS = {"ann", "autoencoder"}


def best_f1_threshold(y_true, scores) -> tuple[float, float]:
    """The score threshold that maximises F1 (predict fraud when score >= threshold), and that F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])


@dataclass
class Prepared:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list
    X_train_sm: np.ndarray = None
    y_train_sm: np.ndarray = None
    info: dict = field(default_factory=dict)


def prepare(train, val, test, seed: int = config.SEED) -> Prepared:
    """Features and scaler fitted on the training rows; SMOTE applied to the (scaled) training rows only."""
    from imblearn.over_sampling import SMOTE

    fb = FeatureBuilder().fit(train)
    F_train, F_val, F_test = fb.transform(train), fb.transform(val), fb.transform(test)
    scaler = StandardScaler().fit(F_train)
    p = Prepared(scaler.transform(F_train), scaler.transform(F_val), scaler.transform(F_test),
                 train["Class"].to_numpy(), val["Class"].to_numpy(), test["Class"].to_numpy(),
                 list(F_train.columns))
    smote = SMOTE(sampling_strategy=0.1, random_state=seed, k_neighbors=5)
    p.X_train_sm, p.y_train_sm = smote.fit_resample(p.X_train, p.y_train)
    p.info = {"n_features": len(p.feature_names),
              "rows": {"train": len(train), "val": len(val), "test": len(test)},
              "fraud": {"train": int(p.y_train.sum()), "val": int(p.y_val.sum()), "test": int(p.y_test.sum())},
              "train_after_smote": {"rows": int(len(p.y_train_sm)), "fraud": int(p.y_train_sm.sum())}}
    return p


# ---------------------------------------------------------------- models: each returns a scoring function

def fit_xgboost(p: Prepared, seed=config.SEED):
    from xgboost import XGBClassifier

    spw = (p.y_train == 0).sum() / (p.y_train == 1).sum()
    model = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=spw, eval_metric="aucpr", random_state=seed, n_jobs=config.N_JOBS,
                          tree_method="hist")
    model.fit(p.X_train_sm, p.y_train_sm, verbose=False)
    return model, lambda X: model.predict_proba(X)[:, 1]


def fit_random_forest(p: Prepared, seed=config.SEED):
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=seed,
                                   n_jobs=config.N_JOBS)
    model.fit(p.X_train_sm, p.y_train_sm)
    return model, lambda X: model.predict_proba(X)[:, 1]


def fit_isolation_forest(p: Prepared, seed=config.SEED):
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(n_estimators=300, contamination=float(p.y_train.mean()), max_samples="auto",
                            random_state=seed, n_jobs=config.N_JOBS)
    model.fit(p.X_train)
    return model, lambda X: -model.score_samples(X)


def fit_lof(p: Prepared, seed=config.SEED):
    from sklearn.neighbors import LocalOutlierFactor

    model = LocalOutlierFactor(n_neighbors=20, contamination=float(p.y_train.mean()), novelty=True,
                               n_jobs=config.N_JOBS)
    model.fit(p.X_train)
    return model, lambda X: -model.score_samples(X)


def _tensorflow(seed):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(config.N_JOBS)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(seed)
    return tf


def fit_autoencoder(p: Prepared, seed=config.SEED):
    """55 -> 64 -> 32 -> 16 -> 32 -> 64 -> 55, trained on legitimate training transactions only."""
    tf = _tensorflow(seed)
    from tensorflow import keras
    from tensorflow.keras import callbacks, layers, regularizers

    dim, reg, dropout = p.X_train.shape[1], regularizers.l2(1e-5), 0.2
    inp = keras.Input(shape=(dim,))
    x = inp
    for units in (64, 32):
        x = layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    x = layers.Dense(16, activation="tanh")(x)  # bounded bottleneck
    for units in (32, 64):
        x = layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)
    out = layers.Dense(dim, activation="linear")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    legit = p.X_train[p.y_train == 0]
    model.fit(legit, legit, epochs=100, batch_size=512, validation_split=0.1, shuffle=True, verbose=0,
              callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
                         callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)])

    def score(X):
        return np.mean((X - model.predict(X, batch_size=4096, verbose=0)) ** 2, axis=1)

    # The notebook searched thresholds on a percentile grid of the training reconstruction errors.
    model.train_legit_mse_ = score(legit)
    return model, score


def fit_ann(p: Prepared, seed=config.SEED):
    """input -> 128 -> 64 -> 32 -> 1 with BatchNorm and Dropout; class weights handle the imbalance."""
    tf = _tensorflow(seed)
    from tensorflow import keras
    from tensorflow.keras import callbacks, layers, regularizers

    reg, dropout = regularizers.l2(1e-4), 0.3
    inp = keras.Input(shape=(p.X_train.shape[1],))
    x = inp
    for units, drop in ((128, dropout), (64, dropout), (32, dropout / 2)):
        x = layers.Dense(units, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(drop)(x)
    model = keras.Model(inp, layers.Dense(1, activation="sigmoid")(x))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(curve="PR", name="auprc")])
    class_weight = {0: 1.0, 1: float((p.y_train == 0).sum() / (p.y_train == 1).sum())}
    # Early stopping watches the validation set; the test set is never used during training.
    model.fit(p.X_train, p.y_train, epochs=100, batch_size=512, validation_data=(p.X_val, p.y_val),
              class_weight=class_weight, shuffle=True, verbose=0,
              callbacks=[callbacks.EarlyStopping(monitor="val_auprc", patience=12, restore_best_weights=True,
                                                 mode="max"),
                         callbacks.ReduceLROnPlateau(monitor="val_auprc", factor=0.5, patience=6, min_lr=1e-7,
                                                     mode="max")])
    return model, lambda X: model.predict(X, batch_size=4096, verbose=0).ravel()


FITTERS = {"xgboost": fit_xgboost, "random_forest": fit_random_forest, "ann": fit_ann,
           "autoencoder": fit_autoencoder, "isolation_forest": fit_isolation_forest, "lof": fit_lof}


def choose_threshold(name, model, y_val, val_scores) -> tuple[float, float]:
    """Decision threshold picked on the validation set: max F1 (autoencoder: over the notebook's percentile grid)."""
    if name == "autoencoder":
        grid = [float(np.percentile(model.train_legit_mse_, q)) for q in np.arange(80, 100, 0.5)]
        f1s = [f1_score(y_val, (val_scores >= t).astype(int), zero_division=0) for t in grid]
        i = int(np.argmax(f1s))
        return grid[i], float(f1s[i])
    return best_f1_threshold(y_val, val_scores)


def run(models=MODELS, seed: int = config.SEED):
    """Fit each model and pick its threshold on validation. Returns (prepared data, {model: result dict})."""
    df = load_raw()
    train, val, test = split(df, seed)
    p = prepare(train, val, test, seed)
    results = {}
    for name in models:
        start = time.time()
        model, score = FITTERS[name](p, seed)
        val_scores, test_scores = score(p.X_val), score(p.X_test)
        thr, val_f1 = choose_threshold(name, model, p.y_val, val_scores)
        results[name] = {"model": model, "val_scores": val_scores, "test_scores": test_scores,
                         "threshold": thr, "val_f1": val_f1, "fit_seconds": round(time.time() - start, 1)}
        print(f"  {name}: threshold {thr:.4f} (validation F1 {val_f1:.4f}), {time.time() - start:.0f}s", flush=True)
    return p, results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    args = parser.parse_args()
    run(args.models)


if __name__ == "__main__":
    main()
