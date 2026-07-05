# ===== models.py (FULLY CORRECTED) =====

import os
import logging
import joblib
import numpy as np
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import ClassifierMixin
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    cross_val_score,
)

import config

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "random_forest": {
        "class": RandomForestClassifier,
        "params": config.RF_PARAMS,
    },
    "svm": {
        "class": CalibratedClassifierCV,
        "params": {
            "estimator": LinearSVC(
                C=1.0, 
                random_state=config.RANDOM_SEED, 
                max_iter=5000
            ),
            "method": "sigmoid",
            "cv": 3,
        },
    },
    "mlp": {
        "class": MLPClassifier,
        "params": config.MLP_PARAMS,
    },
    "xgboost": {
        "class": XGBClassifier,
        "params": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 1,
            "gamma": 0.0,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "eval_metric": "logloss",
            "random_state": config.RANDOM_SEED,
            "n_jobs": -1,
        },
    },
}

def get_model_path(name: str, dataset: str) -> str:
    return os.path.join(config.MODEL_DIR, f"{name}_{dataset}.joblib")

# ─────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────

def build_model(name: str) -> ClassifierMixin:
    """Instantiate a fresh (untrained) model."""
    name = name.lower().strip()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose: {list(MODEL_REGISTRY)}")
    
    entry = MODEL_REGISTRY[name]
    
    if name == "svm":
        base_params = entry["params"].copy()
        estimator = base_params.pop("estimator")
        model = CalibratedClassifierCV(estimator=estimator, **base_params)
    else:
        model = entry["class"](**entry["params"])
    
    logger.debug("Built model '%s'", name)
    return model


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train_model(name: str, X_train: np.ndarray, y_train: np.ndarray) -> ClassifierMixin:
    """Train a single model."""
    logger.info("Training '%s' on %d samples …", name, len(y_train))
    model = build_model(name)
    t0 = time.time()
    model.fit(X_train, y_train)
    # Stamp elapsed seconds so the dashboard Training Time chart can display it
    model.training_time = round(time.time() - t0, 2)
    logger.info("Training complete for '%s' in %.1fs.", name, model.training_time)
    return model


def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Train all registered models."""
    if len(y_train) == 0:
        logger.error("No training data!")
        return {}
    
    trained = {}
    for name in MODEL_REGISTRY:
        try:
            # Run cross-validation first
            cv_result = cross_validate_model(name, X_train, y_train, cv=5)
            
            if cv_result:
                logger.info(
                    "[%s] CV: Acc=%.2f±%.2f | Prec=%.2f | Rec=%.2f | F1=%.2f | AUC=%.4f",
                    name,
                    cv_result.get('accuracy', 0) * 100,
                    cv_result.get('accuracy_std', 0) * 100,
                    cv_result.get('precision', 0) * 100,
                    cv_result.get('recall', 0) * 100,
                    cv_result.get('f1', 0) * 100,
                    cv_result.get('roc_auc', 0),
                )

            # Train full model
            if name == "svm":
                # Use subset for SVM (expensive)
                subset = min(20000, len(X_train))
                logger.info("Training SVM on %d samples …", subset)
                trained[name] = train_model(name, X_train[:subset], y_train[:subset])
            else:
                trained[name] = train_model(name, X_train, y_train)

        except Exception as exc:
            logger.error("Failed to train '%s': %s", name, exc)

    return trained


# ─────────────────────────────────────────────────────────────
# Cross Validation
# ─────────────────────────────────────────────────────────────

def cross_validate_model(name: str, X_train: np.ndarray, y_train: np.ndarray,
                      cv: int = 5) -> dict:
    """Enhanced cross-validation with multiple metrics."""
    if len(y_train) < cv:
        logger.warning("Not enough samples for CV")
        return None
    
    # Choose CV strategy
    if name == "svm":
        cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scoring = ['accuracy', 'precision', 'recall', 'f1']
    else:
        cv_strategy = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=42
        )
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

    try:
        model = build_model(name)
        scores = cross_validate(
            model, X_train, y_train,
            cv=cv_strategy, scoring=scoring, n_jobs=-1
        )
        
        results = {
            'accuracy': scores['test_accuracy'].mean(),
            'accuracy_std': scores['test_accuracy'].std(),
            'precision': scores['test_precision'].mean(),
            'recall': scores['test_recall'].mean(),
            'f1': scores['test_f1'].mean(),
            'roc_auc': np.mean(scores['test_roc_auc'])
                if 'test_roc_auc' in scores else 0,
        }
        
        return results
        
    except Exception as e:
        logger.error("CV failed for %s: %s", name, e)
        return None


def cross_validate_all_models(X_train: np.ndarray, y_train: np.ndarray,
                           cv: int = 5) -> dict:
    """Run CV on all models."""
    results = {}
    for name in MODEL_REGISTRY:
        cv_result = cross_validate_model(name, X_train, y_train, cv)
        if cv_result:
            results[name] = cv_result
    return results


# ─────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────

def predict(model: ClassifierMixin, X: np.ndarray) -> np.ndarray:
    """Binary predictions."""
    return model.predict(X)


def predict_proba(model: ClassifierMixin, X: np.ndarray) -> np.ndarray:
    """Probability estimates for positive class."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        if proba.ndim == 2:
            return proba[:, 1]
        return proba

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return scores
    
    logger.warning("No predict_proba, returning hard labels")
    return model.predict(X).astype(float)


def predict_attack_type(model: ClassifierMixin, X: np.ndarray) -> list:
    """Attack category predictions."""
    preds = model.predict(X)
    attack_map = {
        0: "Normal",
        1: "DoS",
        2: "Probe",
        3: "R2L",
        4: "U2R"
    }
    return [attack_map.get(p, "Unknown") for p in preds]

def hybrid_predict(models: dict, X: np.ndarray):
    """
    Majority voting from all models.
    """
    predictions = []

    for model in models.values():
        predictions.append(model.predict(X))

    predictions = np.array(predictions)

    final_pred = []

    for col in predictions.T:
        values, counts = np.unique(col, return_counts=True)
        final_pred.append(values[np.argmax(counts)])

    return np.array(final_pred)

# ─────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────

def save_model(model: ClassifierMixin, name: str, dataset: str = "nsl_kdd") -> str:
    """Save trained model."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")

    path = get_model_path(name, dataset)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model '%s' saved → %s", name, path)
    return path


def save_all_models(models: dict, dataset: str = "nsl_kdd"):
    """Save all models."""
    for name, model in models.items():
        save_model(model, name, dataset)


def load_model(name: str, dataset: str = "nsl_kdd") -> ClassifierMixin:
    """Load trained model."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")

    path = get_model_path(name, dataset)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model not found: {path}")

    model = joblib.load(path)
    logger.info("Model '%s' loaded ← %s", name, path)
    return model


def load_all_models(dataset: str = "nsl_kdd") -> dict:
    """Load all trained models."""
    loaded = {}
    for name in MODEL_REGISTRY:
        try:
            loaded[name] = load_model(name, dataset)
        except Exception as e:
            logger.warning("Failed to load %s: %s", name, e)
    return loaded



# ─────────────────────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    from data_loader import load_dataset
    from preprocessing import preprocess, save_preprocessor, save_selector

    train_df, test_df = load_dataset()
    X_train, X_test, y_train, y_test, prep, selector = preprocess(
        train_df, test_df
    )
    save_preprocessor(prep)
    save_selector(selector)

    # Cross-validation
    cv_results = cross_validate_all_models(X_train, y_train)
    print("\n=== CV Results ===")
    for name, results in cv_results.items():
        print(f"{name}: Acc={results['accuracy']:.2%} | F1={results['f1']:.2%}")

    # Train
    models = train_all_models(X_train, y_train)
    save_all_models(models)

    # Test predictions
    for name, mdl in models.items():
        preds = predict(mdl, X_test[:5])
        print(f"[{name}] preds: {preds}")