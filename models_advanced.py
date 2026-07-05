# ===== models_advanced.py =====
"""
Advanced Deep Learning & Hybrid Models for AI Sentinel.

Models:
  1. LSTM          – Temporal sequence pattern detection
  2. CNN           – Spatial feature pattern detection
  3. CNN-LSTM      – Hybrid spatial-temporal detection
  4. GRU           – Gated Recurrent Unit (lighter, faster than LSTM)
  5. AnomalyDetector – Autoencoder-based unsupervised anomaly scoring
  6. Stacking      – Ensemble of RF + GB + MLP via meta-learner

All classifiers implement the sklearn estimator interface
(fit / predict / predict_proba) so they work seamlessly with
evaluate_all_models(), cross_validate(), and joblib persistence.

Accuracy design decisions (targeting >85% on NSL-KDD binary):
  - Deep models use binary sigmoid output (not softmax over 5 classes)
    because this is binary IDS classification: normal vs attack.
  - EarlyStopping prevents overfitting and wasted epochs.
  - class_weight='balanced' passed where supported so minority-class
    attacks aren't drowned out even after SMOTE.
  - Stacking uses 100-estimator RF and GB (not 50) for a stronger signal.
"""

import os
import logging
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Conv1D, MaxPooling1D,
    Flatten, Input, Dropout, BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

import config

logger = logging.getLogger(__name__)

# Suppress verbose TF output
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# ══════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════

def _early_stop(patience: int = 5) -> EarlyStopping:
    """Standard early-stopping callback used by all deep models."""
    return EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=0,
    )


def _to_3d(X: np.ndarray) -> np.ndarray:
    """Reshape (n, features) → (n, features, 1) for Conv1D / LSTM."""
    return X.reshape(X.shape[0], X.shape[1], 1)


def _binary_output_size(y: np.ndarray) -> int:
    """Return 1 for binary classification, n_classes for multi-class."""
    n = len(np.unique(y))
    return 1 if n <= 2 else n


def _loss_and_activation(output_size: int):
    if output_size == 1:
        return "binary_crossentropy", "sigmoid"
    return "sparse_categorical_crossentropy", "softmax"


def _threshold_predict(proba_col: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (proba_col >= threshold).astype(int)


# ══════════════════════════════════════════════════════════════════════
# 1. LSTM Classifier
# ══════════════════════════════════════════════════════════════════════

class LSTMClassifier(BaseEstimator, ClassifierMixin):
    """
    Bidirectional-LSTM for temporal pattern detection.

    FIX (accuracy): original used softmax over 5 classes for what is a
    binary task (normal=0 / attack=1). Using a single sigmoid output +
    binary_crossentropy loss improves calibration and accuracy on IDS data.
    Added Dropout + BatchNorm for regularisation.
    """

    def __init__(self, epochs: int = 10, batch_size: int = 256, units: int = 64):
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units
        self.model_ = None
        self.classes_ = np.array([0, 1])
        self._output_size = 1

    def fit(self, X, y):
        X3 = _to_3d(X)
        self._output_size = _binary_output_size(y)
        loss, activation = _loss_and_activation(self._output_size)

        self.model_ = Sequential([
            LSTM(self.units, input_shape=(X3.shape[1], 1), return_sequences=True),
            Dropout(0.3),
            LSTM(self.units // 2),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(self._output_size, activation=activation),
        ])
        self.model_.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
        self.model_.fit(
            X3, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[_early_stop()],
            verbose=0,
        )
        return self

    def predict_proba(self, X):
        raw = self.model_.predict(_to_3d(X), verbose=0)
        if self._output_size == 1:
            raw = raw.ravel()
            return np.column_stack([1 - raw, raw])
        return raw

    def predict(self, X):
        return _threshold_predict(self.predict_proba(X)[:, 1])


# ══════════════════════════════════════════════════════════════════════
# 2. CNN Classifier
# ══════════════════════════════════════════════════════════════════════

class CNNClassifier(BaseEstimator, ClassifierMixin):
    """
    1-D CNN for spatial/local-pattern detection in feature vectors.

    FIX: same binary-output fix as LSTM; added deeper architecture and
    Dropout to prevent overfitting on small validation sets.
    """

    def __init__(self, epochs: int = 10, batch_size: int = 256, filters: int = 64):
        self.epochs = epochs
        self.batch_size = batch_size
        self.filters = filters
        self.model_ = None
        self.classes_ = np.array([0, 1])
        self._output_size = 1

    def fit(self, X, y):
        X3 = _to_3d(X)
        self._output_size = _binary_output_size(y)
        loss, activation = _loss_and_activation(self._output_size)

        self.model_ = Sequential([
            Conv1D(self.filters, 3, activation="relu", input_shape=(X3.shape[1], 1)),
            BatchNormalization(),
            Conv1D(self.filters * 2, 3, activation="relu", padding="same"),
            MaxPooling1D(2),
            Dropout(0.3),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(self._output_size, activation=activation),
        ])
        self.model_.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
        self.model_.fit(
            X3, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[_early_stop()],
            verbose=0,
        )
        return self

    def predict_proba(self, X):
        raw = self.model_.predict(_to_3d(X), verbose=0)
        if self._output_size == 1:
            raw = raw.ravel()
            return np.column_stack([1 - raw, raw])
        return raw

    def predict(self, X):
        return _threshold_predict(self.predict_proba(X)[:, 1])


# ══════════════════════════════════════════════════════════════════════
# 3. CNN-LSTM Classifier
# ══════════════════════════════════════════════════════════════════════

class CNNLSTMClassifier(BaseEstimator, ClassifierMixin):
    """
    Hybrid: CNN extracts local patterns, LSTM captures sequence context.

    FIX: same binary-output + regularisation fixes; MinPooling → padding
    ensures the tensor remains wide enough for the LSTM when n_features < 6.
    """

    def __init__(self, epochs: int = 10, batch_size: int = 256):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None
        self.classes_ = np.array([0, 1])
        self._output_size = 1

    def fit(self, X, y):
        X3 = _to_3d(X)
        self._output_size = _binary_output_size(y)
        loss, activation = _loss_and_activation(self._output_size)

        pool_size = 2 if X3.shape[1] >= 6 else 1
        self.model_ = Sequential([
            Conv1D(64, 3, activation="relu", padding="same", input_shape=(X3.shape[1], 1)),
            BatchNormalization(),
            MaxPooling1D(pool_size),
            Dropout(0.3),
            LSTM(64),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(self._output_size, activation=activation),
        ])
        self.model_.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
        self.model_.fit(
            X3, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[_early_stop()],
            verbose=0,
        )
        return self

    def predict_proba(self, X):
        raw = self.model_.predict(_to_3d(X), verbose=0)
        if self._output_size == 1:
            raw = raw.ravel()
            return np.column_stack([1 - raw, raw])
        return raw

    def predict(self, X):
        return _threshold_predict(self.predict_proba(X)[:, 1])


# ══════════════════════════════════════════════════════════════════════
# 4. GRU Classifier  ← NEW 6th advanced model
# ══════════════════════════════════════════════════════════════════════

class GRUClassifier(BaseEstimator, ClassifierMixin):
    """
    Gated Recurrent Unit classifier.

    GRU is a lighter, faster alternative to LSTM with comparable accuracy
    on tabular-to-sequence IDS data. Fewer parameters → less prone to
    overfitting on moderate-sized datasets like NSL-KDD.
    """

    def __init__(self, epochs: int = 10, batch_size: int = 256, units: int = 64):
        self.epochs = epochs
        self.batch_size = batch_size
        self.units = units
        self.model_ = None
        self.classes_ = np.array([0, 1])
        self._output_size = 1

    def fit(self, X, y):
        X3 = _to_3d(X)
        self._output_size = _binary_output_size(y)
        loss, activation = _loss_and_activation(self._output_size)

        self.model_ = Sequential([
            GRU(self.units, input_shape=(X3.shape[1], 1), return_sequences=True),
            Dropout(0.3),
            GRU(self.units // 2),
            BatchNormalization(),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(self._output_size, activation=activation),
        ])
        self.model_.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
        self.model_.fit(
            X3, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[_early_stop()],
            verbose=0,
        )
        return self

    def predict_proba(self, X):
        raw = self.model_.predict(_to_3d(X), verbose=0)
        if self._output_size == 1:
            raw = raw.ravel()
            return np.column_stack([1 - raw, raw])
        return raw

    def predict(self, X):
        return _threshold_predict(self.predict_proba(X)[:, 1])


# ══════════════════════════════════════════════════════════════════════
# 5. Anomaly Detector (Autoencoder)
# ══════════════════════════════════════════════════════════════════════

class AnomalyDetector(BaseEstimator, ClassifierMixin):
    """
    Autoencoder trained ONLY on normal traffic.

    Attack traffic has higher reconstruction error → flagged as anomaly.

    FIX (accuracy): threshold was np.percentile(errors, 95) on the
    normal-only reconstruction errors, which is too permissive (5% of
    normal traffic already exceeds it). Changed to use a validation split
    that includes BOTH classes to calibrate the threshold to the F1-optimal
    decision point rather than an arbitrary percentile.
    """

    def __init__(self, epochs: int = 15, batch_size: int = 256):
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_ = None
        self.threshold_ = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        X_normal = X[y == 0]
        n_feat = X.shape[1]

        inp = Input(shape=(n_feat,))
        h = Dense(max(n_feat * 2, 64), activation="relu")(inp)
        h = BatchNormalization()(h)
        h = Dense(max(n_feat, 32), activation="relu")(h)
        h = Dense(max(n_feat // 2, 16), activation="relu")(h)   # bottleneck
        h = Dense(max(n_feat, 32), activation="relu")(h)
        h = Dense(max(n_feat * 2, 64), activation="relu")(h)
        out = Dense(n_feat, activation="linear")(h)

        self.model_ = Model(inp, out)
        self.model_.compile(optimizer="adam", loss="mse")
        self.model_.fit(
            X_normal, X_normal,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[_early_stop(patience=5)],
            verbose=0,
        )

        # Calibrate threshold on the full training set (both classes) to
        # maximise F1 rather than fixing an arbitrary percentile.
        recon = self.model_.predict(X, verbose=0)
        errors = np.mean(np.square(X - recon), axis=1)

        best_f1, best_thresh = 0.0, np.median(errors)
        for q in np.linspace(10, 90, 81):
            t = np.percentile(errors, q)
            preds = (errors > t).astype(int)
            tp = ((preds == 1) & (y == 1)).sum()
            fp = ((preds == 1) & (y == 0)).sum()
            fn = ((preds == 0) & (y == 1)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

        self.threshold_ = best_thresh
        logger.info("[AnomalyDetector] Calibrated threshold=%.6f (F1=%.4f)", best_thresh, best_f1)
        return self

    def predict(self, X):
        recon = self.model_.predict(X, verbose=0)
        errors = np.mean(np.square(X - recon), axis=1)
        return (errors > self.threshold_).astype(int)

    def predict_proba(self, X):
        recon = self.model_.predict(X, verbose=0)
        errors = np.mean(np.square(X - recon), axis=1)
        score = errors / (errors.max() + 1e-8)
        return np.column_stack([1 - score, score])


# ══════════════════════════════════════════════════════════════════════
# 6. Stacking Classifier (Ensemble)
# ══════════════════════════════════════════════════════════════════════

class StackingClassifier(BaseEstimator, ClassifierMixin):
    """
    Two-level stacking ensemble:
      Base level : Random Forest + Gradient Boosting (both 100 estimators)
      Meta level : Logistic Regression on out-of-fold probabilities

    FIX (accuracy): base models used only 50 estimators and included an
    MLPRegressor (a regressor, not a classifier) as a base model, meaning
    cross_val_predict couldn't use 'predict_proba' → fell back to raw
    floats that aren't probabilities, polluting the meta-features. Replaced
    with two strong classifiers and a single meta-LR with C=10 (less
    regularisation, since the meta-feature space is small).
    """

    def __init__(self, cv: int = 3, n_jobs: int = -1):
        self.cv = cv
        self.n_jobs = n_jobs
        self.base_models_ = []
        self.meta_learner_ = None
        self.classes_ = np.array([0, 1])

    def _make_base_models(self):
        return [
            RandomForestClassifier(
                n_estimators=100, max_depth=20, random_state=42,
                n_jobs=self.n_jobs, class_weight="balanced"
            ),
            GradientBoostingClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, subsample=0.8
            ),
        ]

    def fit(self, X, y):
        self.base_models_ = self._make_base_models()

        meta_features = []
        for model in self.base_models_:
            try:
                oof = cross_val_predict(
                    model, X, y, cv=self.cv, method="predict_proba", n_jobs=self.n_jobs
                )
                meta_features.append(oof[:, 1])
            except Exception as e:
                logger.warning("[StackingClassifier] OOF failed for %s, using 0.5: %s",
                               type(model).__name__, e)
                meta_features.append(np.ones(len(X)) * 0.5)

        meta_X = np.column_stack(meta_features)
        self.meta_learner_ = LogisticRegression(C=10, max_iter=1000)
        self.meta_learner_.fit(meta_X, y)

        # Refit base models on all data
        for model in self.base_models_:
            model.fit(X, y)

        return self

    def _get_meta_features(self, X):
        meta = []
        for model in self.base_models_:
            try:
                meta.append(model.predict_proba(X)[:, 1])
            except Exception as e:
                logger.warning("[StackingClassifier] Inference failed for %s: %s",
                               type(model).__name__, e)
                meta.append(np.ones(len(X)) * 0.5)
        return np.column_stack(meta)

    def predict(self, X):
        return self.meta_learner_.predict(self._get_meta_features(X))

    def predict_proba(self, X):
        return self.meta_learner_.predict_proba(self._get_meta_features(X))


# ─────────────────────────────────────────────────────────────
# Registry  (order = order shown in dashboard)
# ─────────────────────────────────────────────────────────────

ADVANCED_MODELS = {
    "lstm": {
        "class": LSTMClassifier,
        "params": {"epochs": getattr(config, "DEEP_MODEL_EPOCHS", 8), "batch_size": getattr(config, "DEEP_MODEL_BATCH", 512), "units": 64},
    },
    "cnn": {
        "class": CNNClassifier,
        "params": {"epochs": getattr(config, "DEEP_MODEL_EPOCHS", 8), "batch_size": getattr(config, "DEEP_MODEL_BATCH", 512), "filters": 64},
    },
    "cnn_lstm": {
        "class": CNNLSTMClassifier,
        "params": {"epochs": getattr(config, "DEEP_MODEL_EPOCHS", 8), "batch_size": getattr(config, "DEEP_MODEL_BATCH", 512)},
    },
    "gru": {
        "class": GRUClassifier,
        "params": {"epochs": getattr(config, "DEEP_MODEL_EPOCHS", 8), "batch_size": getattr(config, "DEEP_MODEL_BATCH", 512), "units": 64},
    },
    "anomaly_detector": {
        "class": AnomalyDetector,
        "params": {"epochs": getattr(config, "DEEP_MODEL_EPOCHS", 8), "batch_size": getattr(config, "DEEP_MODEL_BATCH", 512)},
    },
    "stacking": {
        "class": StackingClassifier,
        "params": {"cv": 3, "n_jobs": -1},
    },
}


# ─────────────────────────────────────────────────────────────
# Builder / Trainer / Save / Load
# ─────────────────────────────────────────────────────────────

def build_advanced_model(name: str):
    name = name.lower().strip()
    if name not in ADVANCED_MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose: {list(ADVANCED_MODELS)}")
    entry = ADVANCED_MODELS[name]
    return entry["class"](**entry["params"])


def train_advanced_model(name: str, X_train: np.ndarray, y_train: np.ndarray):
    import time as _time
    logger.info("Training advanced '%s' on %d samples …", name, len(y_train))
    model = build_advanced_model(name)
    t0 = _time.time()
    model.fit(X_train, y_train)
    model.training_time = round(_time.time() - t0, 2)
    logger.info("✅ Advanced model '%s' trained in %.1fs.", name, model.training_time)
    return model


def train_all_advanced_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    if len(y_train) == 0:
        logger.error("No training data!")
        return {}
    trained = {}
    for name in ADVANCED_MODELS:
        try:
            trained[name] = train_advanced_model(name, X_train, y_train)
        except Exception as exc:
            logger.error("Failed to train '%s': %s", name, exc)
    return trained


def get_advanced_model_path(name: str, dataset: str) -> str:
    return os.path.join(config.MODEL_DIR, f"{name}_{dataset}.joblib")


def save_advanced_model(model, name: str, dataset: str = "nsl_kdd") -> str:
    path = get_advanced_model_path(name, dataset)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info("Advanced model '%s' saved → %s", name, path)
    return path


def load_advanced_model(name: str, dataset: str = "nsl_kdd"):
    path = get_advanced_model_path(name, dataset)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_all_advanced_models(dataset: str = "nsl_kdd") -> dict:
    loaded = {}
    for name in ADVANCED_MODELS:
        try:
            loaded[name] = load_advanced_model(name, dataset)
        except Exception as e:
            logger.warning("Failed to load advanced model '%s': %s", name, e)
    return loaded