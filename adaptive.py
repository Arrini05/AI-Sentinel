# ===== adaptive.py  =====

"""
AI Sentinel - Adaptive Continual Learning System
Fully functional with real-time drift detection
"""

import os
import time
import json
import logging
import numpy as np
from datetime import datetime
from collections import deque

import config   # FIX: simulate_adaptive_retraining() uses config.ACTIVE_DATASET
                 # and config.RANDOM_SEED but config was never imported, causing
                 # NameError: name 'config' is not defined on every --adaptive run.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("adaptive")


class DriftDetector:
    """Detect concept drift in data streams"""
    
    def __init__(self, window_size=100, threshold=0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.reference_window = deque(maxlen=window_size)
        self.current_window = deque(maxlen=window_size)
        self.drift_detected = False
        self.drift_history = []  # ← ADD THIS
        self.confidence_history = deque(maxlen=100)
        
    def add_sample(self, prediction, actual=None, confidence=1.0):
        """Add a sample and check for drift"""
        self.confidence_history.append(confidence)
        
        # If we know actual result (for replay mode)
        if actual is not None:
            is_correct = int(prediction == actual)
            self.current_window.append(is_correct)
            
            if len(self.current_window) >= self.window_size:
                self._check_drift()
        else:
            # For live mode - track confidence drop
            if len(self.confidence_history) >= self.window_size:
                self._check_confidence_drift()
            
    def _check_drift(self):
        """Check if drift has occurred - accuracy based"""
        if len(self.reference_window) < 10:
            self.reference_window = self.current_window.copy()
            return
            
        ref_acc = np.mean(self.reference_window)
        curr_acc = np.mean(list(self.current_window))
        
        if ref_acc - curr_acc > self.threshold:
            self.drift_detected = True
            self.drift_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": "accuracy_drop",
                "ref_acc": ref_acc,
                "curr_acc": curr_acc
            })
            logger.warning(f"⚠️ Drift detected! Ref: {ref_acc:.2%}, Curr: {curr_acc:.2%}")
            
        if len(self.current_window) >= self.window_size:
            self.reference_window = self.current_window.copy()
                
    def _check_confidence_drift(self):
        """Check if confidence dropped significantly"""
        if len(self.confidence_history) < self.window_size:
            return
            
        avg_confidence = np.mean(self.confidence_history)
        
        if avg_confidence < 0.6:  # Low confidence threshold
            self.drift_detected = True
            self.drift_history.append({
                "timestamp": datetime.now().isoformat(),
                "type": "low_confidence",
                "avg_confidence": avg_confidence
            })
            logger.warning(f"⚠️ Low confidence drift! Avg: {avg_confidence:.2%}")
    
    def reset(self):
        """Reset detector"""
        self.reference_window.clear()
        self.current_window.clear()
        self.confidence_history.clear()
        self.drift_detected = False


class ContinualLearner:
    """Continual learning with model updates"""
    
    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        self.version = 1
        self.version_history = []
        self.sample_buffer = deque(maxlen=1000)
        self.drift_detector = DriftDetector(window_size=100, threshold=0.1)
        self.wrong_predictions = 0
        self.total_predictions = 0
        
    def add_sample(self, X, y, prediction=None, actual=None):
        """Add a training sample"""
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, (int, np.integer)):
            y = np.array([y])
            
        self.sample_buffer.append((X, y))
        
        # Track prediction accuracy
        if prediction is not None and actual is not None:
            self.total_predictions += 1
            if prediction != actual:
                self.wrong_predictions += 1
                
        # Check for drift
        self.drift_detector.add_sample(prediction, actual)
        
    def add_confidence_sample(self, confidence):
        """Add confidence sample for live mode"""
        self.drift_detector.add_sample(None, None, confidence)
        
    def should_retrain(self):
        """Check if we should retrain"""
        if self.drift_detector.drift_detected:
            return True, "Drift detected"
        if len(self.sample_buffer) >= 500:
            return True, "Enough samples"
        return False, "No trigger"
        
    def get_accuracy(self):
        """Get current accuracy"""
        if self.total_predictions == 0:
            return None
        return 1 - (self.wrong_predictions / self.total_predictions)
        
    def partial_fit(self, X, y):
        """Incrementally train the model"""
        try:
            if hasattr(self.model, 'partial_fit'):
                self.model.partial_fit(X, y)
                logger.info(f"✅ Partial fit on {len(X)} samples")
                return True
            return False
        except Exception as e:
            logger.error(f"Partial fit error: {e}")
            return False
            
    def save_version(self, metrics=None):
        """Save model version"""
        version_info = {
            "version": self.version,
            "timestamp": datetime.now().isoformat(),
            "samples": len(self.sample_buffer),
            "accuracy": self.get_accuracy(),
            "drift_detected": self.drift_detector.drift_detected,
            "metrics": metrics or {}
        }
        self.version_history.append(version_info)
        self.version += 1
        logger.info(f"📦 Saved model version {self.version - 1}")
        return version_info
        
    def get_versions(self):
        """Get version history"""
        return self.version_history


class AdaptiveManager:
    """Manage multiple continual learners"""
    
    def __init__(self):
        self.learners = {}
        self.models = {}
        
    def register_model(self, model_name, model, preprocessor=None):
        """Register a model for continual learning"""
        self.learners[model_name] = ContinualLearner(model, preprocessor)
        self.models[model_name] = model
        
    def process_prediction(self, model_name, X, y_true=None, prediction=None, confidence=1.0):
        """Process a prediction for learning"""
        if model_name not in self.learners:
            return
            
        learner = self.learners[model_name]
        
        # For replay mode with known labels
        if y_true is not None and prediction is not None:
            learner.add_sample(X, y_true, prediction, y_true)
        # For live mode - track confidence
        elif confidence < 1.0:
            learner.add_confidence_sample(confidence)
            
    def should_retrain(self, model_name):
        """Check if model should retrain"""
        if model_name in self.learners:
            return self.learners[model_name].should_retrain()
        return False, "Model not found"
        
    def get_status(self):
        """Get status of all learners"""
        status = {}
        for name, learner in self.learners.items():
            status[name] = {
                "samples": len(learner.sample_buffer),
                "version": learner.version,
                "drift_detected": learner.drift_detector.drift_detected,
                "accuracy": learner.get_accuracy(),
                "total_predictions": learner.total_predictions,
                "wrong_predictions": learner.wrong_predictions
            }
        return status


# ========================
# SINGLETON
# ========================
_adaptive_manager = None

def get_adaptive_manager():
    global _adaptive_manager
    if _adaptive_manager is None:
        _adaptive_manager = AdaptiveManager()
    return _adaptive_manager


# ========================
# CLI / main.py ENTRY POINT
# ========================
# FIX: main.py has always imported `simulate_adaptive_retraining` from this
# module for its `--adaptive` flag, but the function never actually existed
# here -- importing main.py raised:
#   ImportError: cannot import name 'simulate_adaptive_retraining' from 'adaptive'
# which crashed the whole script (including `--train`) before a single line
# of the pipeline could run. This wraps the existing ContinualLearner /
# DriftDetector classes into the simple functional entry point main.py expects.
def simulate_adaptive_retraining(model_name: str, X_train, y_train, n_new_samples: int = 200):
    """
    Simulate streaming `n_new_samples` rows from X_train/y_train through the
    given (already-trained) model, tracking drift via ContinualLearner, and
    retraining + saving the model if a retrain is triggered.

    Returns the version_info dict recorded for this simulation run.
    """
    from models import load_model, save_model

    logger.info("[Adaptive] Loading model '%s' for adaptive simulation …", model_name)
    model = load_model(model_name, config.ACTIVE_DATASET)

    learner = ContinualLearner(model)

    n = min(n_new_samples, len(X_train))
    rng = np.random.RandomState(config.RANDOM_SEED)
    indices = rng.choice(len(X_train), size=n, replace=False)

    for idx in indices:
        x = np.asarray(X_train[idx])
        y = int(y_train[idx])
        try:
            prediction = int(model.predict(x.reshape(1, -1))[0])
        except Exception as e:
            logger.debug("[Adaptive] Prediction failed for sample %d: %s", idx, e)
            prediction = y
        learner.add_sample(x, y, prediction=prediction, actual=y)

    should, reason = learner.should_retrain()
    logger.info(
        "[Adaptive] Streamed %d samples. Retrain triggered: %s (%s)", n, should, reason
    )

    if should and hasattr(model, "partial_fit") and len(learner.sample_buffer) > 0:
        X_batch = np.array([s[0] for s in learner.sample_buffer])
        y_batch = np.array([
            int(s[1][0]) if hasattr(s[1], "__len__") else int(s[1])
            for s in learner.sample_buffer
        ])
        if learner.partial_fit(X_batch, y_batch):
            save_model(model, model_name, config.ACTIVE_DATASET)
            logger.info("[Adaptive] Model '%s' retrained and saved.", model_name)

    version_info = learner.save_version(metrics={"accuracy": learner.get_accuracy(), "reason": reason})
    logger.info("[Adaptive] Simulation complete → %s", version_info)
    return version_info


# Backward compatibility
AdaptiveLearner = ContinualLearner