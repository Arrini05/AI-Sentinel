# ===== attack_classifier.py (FULLY CORRECTED) =====

import os
import logging
from typing import Optional, List, Dict

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

import config

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────

ATTACK_CLF_PATH = os.path.join(config.MODEL_DIR, "attack_type_clf_{dataset}.joblib")


def _clf_path(dataset: str) -> str:
    return ATTACK_CLF_PATH.format(dataset=dataset)


# ──────────────────────────────────────────────────────────────
# Label Mapping Helpers
# ──────────────────────────────────────────────────────────────

def map_nsl_label_to_class(label: str) -> int:
    """Map raw NSL-KDD string label → 0-4 integer class."""
    return config.NSL_KDD_ATTACK_FAMILIES.get(label.strip().lower(), 1)


def class_to_attack_name(cls: int) -> str:
    """Convert class ID (0-4) to attack name."""
    return config.ATTACK_TYPES.get(int(cls), "Unknown")


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train_attack_classifier(
    X_train: np.ndarray,
    y_multiclass: np.ndarray,
    dataset: str = "nsl_kdd"
) -> RandomForestClassifier:
    """Train and save a multi-class attack type classifier."""
    n_classes = len(set(y_multiclass))
    
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=25,
        min_samples_split=4,
        random_state=config.RANDOM_SEED,
        n_jobs=-1,
        class_weight="balanced",
    )
    
    logger.info("[AttackCLF] Training on %d samples, %d classes …",
              len(y_multiclass), n_classes)
    
    clf.fit(X_train, y_multiclass)
    _save_classifier(clf, dataset)
    
    logger.info("[AttackCLF] Training complete.")
    return clf


def _save_classifier(clf: RandomForestClassifier, dataset: str) -> None:
    """Save classifier to disk."""
    path = _clf_path(dataset)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(clf, path)
    logger.info("[AttackCLF] Saved → %s", path)


def load_attack_classifier(dataset: str = "nsl_kdd") -> Optional[RandomForestClassifier]:
    """Load pre-trained attack type classifier."""
    path = _clf_path(dataset)
    
    if not os.path.isfile(path):
        logger.warning("[AttackCLF] Not found: %s – fallback will be used.", path)
        return None
    
    try:
        clf = joblib.load(path)
        logger.info("[AttackCLF] Loaded ← %s", path)
        return clf
    except Exception as e:
        logger.error("[AttackCLF] Failed to load %s: %s", path, e)
        return None


# ──────────────────────────────────────────────────────────────
# Heuristic Fallback
# ──────────────────────────────────────────────────────────────

def _heuristic_attack_type(features: np.ndarray) -> str:
    """
    Rule-based fallback when multi-class model is unavailable.
    Uses feature thresholds to guess attack type.
    """
    x = features.flatten()
    n = len(x)

    # Extract features (index positions depend on preprocessing)
    duration = float(x[0]) if n > 0 else 0.0
    src_bytes = float(x[4]) if n > 4 else 0.0
    dst_bytes = float(x[5]) if n > 5 else 0.0
    count = float(x[22]) if n > 22 else 0.0

    # Rule-based classification
    if count > 1.5:
        return "DoS"
    elif duration < -0.5 and src_bytes < 0.0:
        return "Probe"
    elif src_bytes > 1.0 and dst_bytes < 0.1:
        return "R2L"
    elif abs(src_bytes - dst_bytes) > 2.0:
        return "U2R"
    else:
        return "DoS"


# ──────────────────────────────────────────────────────────────
# Two-Stage Inference
# ──────────────────────────────────────────────────────────────

class AttackTypeClassifier:
    """
    Two-stage pipeline:
      1. Binary model decides Normal vs Attack
      2. Multi-class model identifies attack family
    """

    def __init__(
        self,
        binary_model,
        multiclass_model: Optional[RandomForestClassifier] = None
    ):
        self.binary_model = binary_model
        self.multiclass_model = multiclass_model
        
        logger.info("[AttackTypeClassifier] Initialized with multi-class model: %s",
                  "Yes" if multiclass_model else "No (heuristic fallback)")

    def predict(self, X: np.ndarray) -> List[Dict]:
        """Run two-stage prediction on a batch."""
        binary_preds = self.binary_model.predict(X)

        # Get binary probabilities
        if hasattr(self.binary_model, "predict_proba"):
            binary_proba = self.binary_model.predict_proba(X)[:, 1]
        else:
            binary_proba = binary_preds.astype(float)

        results = []
        
        for i, (pred, prob) in enumerate(zip(binary_preds, binary_proba)):
            if pred == 0:
                # Normal traffic
                results.append({
                    "binary_pred": 0,
                    "attack_type": "Normal",
                    "confidence": round(1.0 - float(prob), 4),
                    "class_id": 0,
                })
            else:
                # Attack detected - Stage 2 classification
                x_row = X[i:i+1]
                
                if self.multiclass_model is not None:
                    try:
                        cls = int(self.multiclass_model.predict(x_row)[0])
                        mc_proba = self.multiclass_model.predict_proba(x_row)[0]
                        conf = round(float(mc_proba[cls]), 4)
                    except Exception as e:
                        logger.warning("[AttackCLF] Prediction failed, using heuristic: %s", e)
                        attack_name = _heuristic_attack_type(x_row)
                        cls = {v: k for k, v in config.ATTACK_TYPES.items()}.get(attack_name, 1)
                        conf = round(float(prob), 4)
                else:
                    # Use heuristic fallback
                    attack_name = _heuristic_attack_type(x_row)
                    cls = {v: k for k, v in config.ATTACK_TYPES.items()}.get(attack_name, 1)
                    conf = round(float(prob), 4)

                results.append({
                    "binary_pred": 1,
                    "attack_type": class_to_attack_name(cls),
                    "confidence": conf,
                    "class_id": cls,
                })

        return results

    def predict_single(self, x: np.ndarray) -> Dict:
        """Convenience wrapper for a single sample."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.predict(x)[0]