# ===== smart_model_selector.py (FULLY CORRECTED) =====

"""
Smart model selection for AI Sentinel.
Automatically switches between models based on traffic conditions.
"""

import logging
import time
from collections import deque
from typing import Dict

import numpy as np

import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# SmartModelSelector
# ──────────────────────────────────────────────────────────────

class SmartModelSelector:
    """
    Automatically selects the best model for current traffic conditions.
    
    Parameters
    ----------
    models      : dict   model_name → fitted sklearn estimator
    window_size : int    rolling window for packet-rate calculation
    """

    # Reason strings for logging
    _REASON = {
        "random_forest": "⚡ High packet rate → Random Forest selected (fastest inference)",
        "svm": "🎯 Low feature variance → SVM selected (tight decision boundary)",
        "mlp": "🧠 Complex traffic pattern → MLP selected (deep non-linear model)",
    }

    def __init__(self, models: dict, window_size: int = 50):
        if not models:
            raise ValueError("No models supplied to SmartModelSelector.")
        
        self.models = models
        self.window_size = window_size
        
        # Rolling windows
        self._timestamps: deque = deque(maxlen=window_size)
        self._conf_window: deque = deque(maxlen=window_size)
        
        # Default model
        self._current_model_name: str = self._pick_default()
        self._last_switch_reason: str = "Initial selection"
        self._switch_count: int = 0
        self._last_rate = 0
        self._stable_counter = 0
        
        logger.info("[SmartSelector] Initialised. Default: %s", self._current_model_name)

    # ── Properties ──────────────────────────────────────────────
    @property
    def current_model(self):
        """Return active fitted model."""
        return self.models[self._current_model_name]

    @property
    def current_model_name(self) -> str:
        """Return name of active model."""
        return self._current_model_name

    @property
    def switch_reason(self) -> str:
        """Return reason for last model switch."""
        return self._last_switch_reason

    @property
    def switch_count(self) -> int:
        """Return number of model switches."""
        return self._switch_count

    # ── Public API ──────────────────────────────────────────────
    def predict(self, X: np.ndarray):
        """Record timestamp and run inference."""
        self._timestamps.append(time.time())
        self._maybe_switch(X)

        model = self.current_model
        preds = model.predict(X)
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
        else:
            probs = preds.astype(float).reshape(-1, 1)

        # Track confidence
        if probs.ndim == 2:
            conf = float(np.max(probs, axis=1).mean())
        else:
            conf = float(np.mean(probs))
        self._conf_window.append(conf)

        return preds, probs, self._current_model_name, self._last_switch_reason

    def summary(self) -> Dict:
        """Return selector statistics."""
        return {
            "current_model": self._current_model_name,
            "packet_rate": self._packet_rate(),
            "avg_confidence": (
                sum(self._conf_window) / len(self._conf_window)
                if self._conf_window else 0.0
            ),
            "switch_count": self._switch_count,
            "reason": self._last_switch_reason,
        }

    # ── Internal Helpers ────────────────────────────────────────
    def _pick_default(self) -> str:
        """Pick default model."""
        if "random_forest" in self.models:
            return "random_forest"
        return next(iter(self.models))

    def _packet_rate(self) -> float:
        """Calculate packets per second."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return len(self._timestamps) / elapsed

    def _feature_variance(self, X: np.ndarray) -> float:
        """Calculate mean feature variance."""
        if X.size == 0:
            return 1.0
        return float(np.mean(np.var(X, axis=0))) if X.shape[0] > 1 else 1.0

    def _maybe_switch(self, X: np.ndarray) -> None:
        """Switch model based on traffic conditions."""
        rate = self._packet_rate()
        variance = self._feature_variance(X)
    
        if abs(rate - self._last_rate) < 0.5:
            self._stable_counter += 1
        else:
            self._stable_counter = 0

        self._last_rate = rate

        if self._stable_counter < 3:
            return

        # Selection rules
        if rate > config.SMART_MODEL_HIGH_RATE_THRESHOLD and "random_forest" in self.models:
            candidate = "random_forest"
        elif variance < config.SMART_MODEL_LOW_VARIANCE_THRESHOLD and "svm" in self.models:
            candidate = "svm"
        elif "mlp" in self.models and rate > 5:
            candidate = "mlp"
        else:
            candidate = self._current_model_name

        # Switch if needed
        if candidate != self._current_model_name:
            self._current_model_name = candidate
            self._last_switch_reason = self._REASON.get(candidate, candidate)
            self._switch_count += 1
            logger.info(
                "[SmartSelector] Switched → %s (rate=%.1f pps, var=%.4f)",
                candidate, rate, variance
            )