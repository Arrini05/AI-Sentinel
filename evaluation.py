# ===== evaluation.py (FULLY CORRECTED) =====

import logging
import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    classification_report,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.base import ClassifierMixin

import config
from models import predict, predict_proba

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Chart Style
# ─────────────────────────────────────────────────────────────

CHART_STYLE = {
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#cbd5e1",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "text.color": "#f1f5f9",
    "grid.color": "#334155",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
}

def _apply_style() -> None:
    plt.rcParams.update(CHART_STYLE)


# ─────────────────────────────────────────────────────────────
# Core Metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                 y_scores: np.ndarray = None) -> dict:
    """Compute binary classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    roc_auc = 0.0
    if y_scores is not None:
        try:
            roc_auc = auc(*roc_curve(y_true, y_scores)[:2])
        except Exception:
            roc_auc = 0.0

    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0) * 100, 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0) * 100, 4),
        "false_positive_rate": round(fpr * 100, 4),
        "auc": round(roc_auc, 4),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def evaluate_model(model: ClassifierMixin, X_test: np.ndarray, y_test: np.ndarray,
                model_name: str = "model") -> dict:
    """Evaluate a single model."""
    y_pred = predict(model, X_test)
    y_scores = predict_proba(model, X_test) if hasattr(model, "predict_proba") else None
    metrics = compute_metrics(y_test, y_pred, y_scores)

    logger.info(
        "[%s] Acc=%.2f%% | Prec=%.2f%% | Rec=%.2f%% | F1=%.2f%% | FPR=%.2f%% | AUC=%.4f",
        model_name, metrics["accuracy"], metrics["precision"], metrics["recall"],
        metrics["f1"], metrics["false_positive_rate"], metrics["auc"],
    )
    return metrics


def evaluate_all_models(models: dict, X_test: np.ndarray, 
                     y_test: np.ndarray) -> dict:
    """Evaluate all models."""
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test, model_name=name)
    return results


def print_classification_report(model: ClassifierMixin, X_test: np.ndarray,
                        y_test: np.ndarray, model_name: str = "model") -> None:
    """Print sklearn classification report."""
    y_pred = predict(model, X_test)
    print(f"\n{'─'*55}")
    print(f"Classification Report – {model_name}")
    print(f"{'─'*55}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"], zero_division=0))


# ─────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                        model_name: str = "Model",
                        save_path: Optional[str] = None) -> plt.Figure:
    """Plot confusion matrix."""
    _apply_style()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = [["TN", "FP"], ["FN", "TP"]]
    class_names = ["Normal", "Attack"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12, labelpad=8)
    ax.set_ylabel("Actual", fontsize=12, labelpad=8)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13, pad=12, color="#f1f5f9")

    thresh = cm.max() / 2.0
    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{labels[r][c]}\n{cm[r, c]:,}", ha="center", va="center",
                    fontsize=12, color="white" if cm[r, c] > thresh else "#0f172a")

    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info("CM saved → %s", save_path)

    plt.close(fig)
    return fig


def plot_roc_curve(model: ClassifierMixin, X_test: np.ndarray, y_test: np.ndarray,
                model_name: str = "Model",
                save_path: Optional[str] = None) -> plt.Figure:
    """Plot ROC curve."""
    _apply_style()

    if not hasattr(model, "predict_proba"):
        logger.warning("ROC skipped: no predict_proba")
        return None

    y_scores = predict_proba(model, X_test)
    fpr_vals, tpr_vals, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr_vals, tpr_vals)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr_vals, tpr_vals, color="#38bdf8", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve – {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120)
        logger.info("ROC saved → %s", save_path)

    plt.close(fig)
    return fig


def plot_precision_recall_curve(model: ClassifierMixin, X_test: np.ndarray,
                              y_test: np.ndarray, model_name: str = "Model",
                              save_path: Optional[str] = None) -> plt.Figure:
    """Plot Precision-Recall curve."""
    _apply_style()
    
    if not hasattr(model, "predict_proba"):
        logger.warning("PR skipped: no predict_proba")
        return None
    
    y_scores = predict_proba(model, X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    ap = average_precision_score(y_test, y_scores)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color="#f87171", lw=2, label=f"AP = {ap:.4f}")
    ax.plot([0, 1], [1, 0], color="#64748b", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve – {model_name}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120)
    
    plt.close(fig)
    return fig


def plot_bootstrap_roc(model: ClassifierMixin, X_test: np.ndarray, y_test: np.ndarray,
                     n_bootstrap: int = 1000, model_name: str = "Model",
                     save_path: Optional[str] = None) -> plt.Figure:
    """Plot ROC with bootstrap confidence intervals."""
    _apply_style()
    
    if not hasattr(model, "predict_proba"):
        return None
    
    y_scores = predict_proba(model, X_test)
    auc_scores = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
        if len(set(y_test[indices])) < 2:
            continue
        try:
            auc_scores.append(roc_auc_score(y_test[indices], y_scores[indices]))
        except ValueError:
            # Expected when a bootstrap resample contains only one class.
            continue
    
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#38bdf8", lw=2, label=f"AUC = {roc_auc:.4f}")
    
    if auc_scores:
        lower = np.percentile(auc_scores, 2.5)
        upper = np.percentile(auc_scores, 97.5)
        ax.fill_between([0], [0], [1], color="#38bdf8", alpha=0.1,
                    label=f"95% CI: [{lower:.3f}, {upper:.3f}]")
    
    ax.plot([0, 1], [0, 1], color="#64748b", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC (Bootstrap) – {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120)
    
    plt.close(fig)
    return fig


def plot_class_distribution(y: np.ndarray, title: str = "Class Distribution",
                         save_path: Optional[str] = None) -> plt.Figure:
    """Plot class distribution."""
    _apply_style()
    labels = ["Normal (0)", "Attack (1)"]
    counts = [(y == 0).sum(), (y == 1).sum()]
    colors = ["#34d399", "#f87171"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="#334155", linewidth=0.8)
    ax.bar_label(bars, labels=[f"{c:,}" for c in counts], padding=4, color="#f1f5f9", fontsize=11)
    ax.set_title(title, fontsize=13, pad=10, color="#f1f5f9")
    ax.set_ylabel("Count", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(axis="y")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info("Distribution saved → %s", save_path)

    plt.close(fig)
    return fig


def plot_metrics_comparison(results: dict, save_path: Optional[str] = None) -> plt.Figure:
    """Compare metrics across models."""
    _apply_style()
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metric_keys)

    x = np.arange(n_metrics)
    width = 0.22
    palette = ["#38bdf8", "#34d399", "#fb923c", "#c084fc"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i, (name, color) in enumerate(zip(model_names, palette)):
        vals = [results[name].get(k, 0) for k in metric_keys]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name.replace("_", " ").title(),
                    color=color, edgecolor="#0f172a", linewidth=0.5)
        ax.bar_label(bars, labels=[f"{v:.1f}" for v in vals], padding=2, fontsize=7.5, color="#f1f5f9")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("Model Comparison", fontsize=13, pad=10, color="#f1f5f9")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.3)
    ax.grid(axis="y")
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info("Comparison saved → %s", save_path)

    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    from data_loader import load_dataset
    from preprocessing import preprocess
    from models import train_all_models

    train_df, test_df = load_dataset()
    X_train, X_test, y_train, y_test, _, _ = preprocess(train_df, test_df)
    models = train_all_models(X_train, y_train)

    results = evaluate_all_models(models, X_test, y_test)
    for name, m in results.items():
        print(f"\n{name}: {m}")

    # Create charts directory
    charts_dir = os.path.join(config.LOG_DIR, "charts")
    os.makedirs(charts_dir, exist_ok=True)

    # Save all charts
    for name, model in models.items():
        y_pred = predict(model, X_test)
        
        # Confusion matrix
        plot_confusion_matrix(
            y_test, y_pred, name,
            save_path=os.path.join(charts_dir, f"cm_{name}.png")
        )
        
        # ROC curve
        plot_roc_curve(
            model, X_test, y_test, name,
            save_path=os.path.join(charts_dir, f"roc_{name}.png")
        )
        
        # Precision-Recall curve
        plot_precision_recall_curve(
            model, X_test, y_test, name,
            save_path=os.path.join(charts_dir, f"pr_{name}.png")
        )
        
        print(f"Charts saved for {name}")

    # Class distribution
    plot_class_distribution(
        y_test,
        save_path=os.path.join(charts_dir, "class_distribution.png")
    )
    
    # Metrics comparison
    plot_metrics_comparison(
        results,
        save_path=os.path.join(charts_dir, "model_comparison.png")
    )
    
    print(f"\nAll charts saved to {charts_dir}")