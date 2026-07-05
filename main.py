# ===== main.py (FULLY CORRECTED) =====

import argparse
import logging
import os
import sys

import config
from data_loader import load_dataset
from preprocessing import (
    preprocess,
    save_preprocessor,
    load_preprocessor,
    save_selector,
)
from models import (
    train_all_models,
    save_all_models,
    load_all_models,
    predict,
    predict_proba,
    MODEL_REGISTRY,
)
from evaluation import (
    evaluate_all_models,
    print_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_class_distribution,
    plot_metrics_comparison,
)
from adaptive import simulate_adaptive_retraining
from alerts import AlertManager, generate_fake_metadata

import numpy as np
from models_advanced import (
    train_all_advanced_models,
    save_advanced_model,
    load_all_advanced_models,
    ADVANCED_MODELS,
)

# ─────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────

os.makedirs(config.LOG_DIR, exist_ok=True)
os.makedirs(config.ALERT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(config.LOG_DIR, "sentinel.log"),
            mode="w",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────
# Pipeline Steps
# ─────────────────────────────────────────────────────────────

def run_training(dataset: str = config.ACTIVE_DATASET) -> dict:
    """Full training pipeline: load → preprocess → train → save."""
    logger.info("="*50)
    logger.info("  TRAINING PIPELINE  –  dataset: %s", dataset)
    logger.info("="*50)

    # Step 1: Load
    logger.info("Step 1/4 – Loading dataset …")
    train_df, test_df = load_dataset(dataset)

    # Step 2: Preprocess
    logger.info("Step 2/4 – Preprocessing …")
    X_train, X_test, y_train, y_test, preprocessor, selector = preprocess(
        train_df, test_df, dataset
    )

    save_preprocessor(preprocessor, dataset)
    save_selector(selector, dataset)

    # Step 3: Train base models
    logger.info("Step 3/5 – Training base models (RF, SVM, MLP, XGBoost) …")
    models = train_all_models(X_train, y_train)

    # Step 4: Save base models
    logger.info("Step 4/5 – Saving base models …")
    save_all_models(models, dataset)

    # Step 5: Train and save advanced models
    # FIX: train_all_advanced_models was imported but never called, so
    # `python main.py --train` only ever produced 4 base model files. The
    # Load button in the dashboard then loaded 0 advanced models every time.
    logger.info("Step 5/5 – Training advanced models (LSTM, CNN, CNN-LSTM, Anomaly, Stacking) …")
    cap = getattr(config, "MAX_TRAIN_SAMPLES", 20000)
    if len(X_train) > cap:
        rng_adv = np.random.RandomState(config.RANDOM_SEED)
        idx_adv = rng_adv.choice(len(X_train), size=cap, replace=False)
        X_adv, y_adv = X_train[idx_adv], y_train[idx_adv]
    else:
        X_adv, y_adv = X_train, y_train

    advanced_models = {}
    for adv_name in ADVANCED_MODELS:
        try:
            from models_advanced import train_advanced_model as _train_adv_one
            adv_m = _train_adv_one(adv_name, X_adv, y_adv)
            save_advanced_model(adv_m, adv_name, dataset)
            advanced_models[adv_name] = adv_m
            logger.info("✅ Advanced model '%s' trained and saved.", adv_name)
        except Exception as exc:
            logger.warning("⚠️  Advanced model '%s' failed: %s", adv_name, exc)

    models.update(advanced_models)
    logger.info(
        "Training complete. %d base + %d advanced = %d total model(s) saved.",
        len(models) - len(advanced_models), len(advanced_models), len(models)
    )

    return {
        "models": models,
        "preprocessor": preprocessor,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


def run_evaluation(dataset: str = config.ACTIVE_DATASET,
                  save_charts: bool = True) -> dict:
    """Evaluate trained models on test set."""
    logger.info("="*50)
    logger.info("  EVALUATION PIPELINE  –  dataset: %s", dataset)
    logger.info("="*50)

    # Load models
    models = load_all_models(dataset)
    if not models:
        logger.error("No trained models found. Run --train first.")
        sys.exit(1)

    # Load data
    train_df, test_df = load_dataset(dataset)
    preprocessor = load_preprocessor(dataset)

    X_train, X_test, y_train, y_test, _, _ = preprocess(
        train_df, test_df, dataset
    )

    # Evaluate
    results = evaluate_all_models(models, X_test, y_test)

    # Print reports
    for name, model in models.items():
        print_classification_report(model, X_test, y_test, name)

    # Save charts
    if save_charts:
        charts_dir = os.path.join(config.LOG_DIR, "charts")
        os.makedirs(charts_dir, exist_ok=True)

        for name, model in models.items():
            y_pred = predict(model, X_test)
            plot_confusion_matrix(
                y_test, y_pred, name,
                save_path=os.path.join(charts_dir, f"cm_{name}.png"),
            )
            plot_roc_curve(
                model, X_test, y_test, name,
                save_path=os.path.join(charts_dir, f"roc_{name}.png"),
            )

        plot_class_distribution(
            y_test,
            save_path=os.path.join(charts_dir, "class_distribution.png"),
        )
        plot_metrics_comparison(
            results,
            save_path=os.path.join(charts_dir, "model_comparison.png"),
        )
        logger.info("Charts saved to %s", charts_dir)

    _print_summary_table(results)
    return results


def run_adaptive(model_name: str,
               dataset: str = config.ACTIVE_DATASET,
               n_new_samples: int = 200) -> None:
    """Simulate adaptive retraining."""
    logger.info("="*50)
    logger.info("  ADAPTIVE RETRAINING  –  model: %s", model_name)
    logger.info("="*50)

    if model_name not in MODEL_REGISTRY:
        logger.error("Unknown model '%s'. Choose from: %s",
                   model_name, list(MODEL_REGISTRY))
        sys.exit(1)

    train_df, test_df = load_dataset(dataset)
    X_train, X_test, y_train, y_test, _, _ = preprocess(
        train_df, test_df, dataset
    )

    simulate_adaptive_retraining(model_name, X_train, y_train, n_new_samples)
    logger.info("Adaptive simulation complete.")


def run_alert_demo(dataset: str = config.ACTIVE_DATASET,
                 n_samples: int = 50) -> None:
    """Run alert demo."""
    logger.info("="*50)
    logger.info("  ALERT DEMO  –  %d samples", n_samples)
    logger.info("="*50)

    models = load_all_models(dataset)
    if not models:
        logger.error("No models loaded. Run --train first.")
        sys.exit(1)

    train_df, test_df = load_dataset(dataset)
    _, X_test, _, y_test, _, _ = preprocess(train_df, test_df, dataset)

    mgr = AlertManager()
    model = next(iter(models.values()))
    model_name = next(iter(models))

    X_batch = X_test[:n_samples]
    preds = predict(model, X_batch)
    probs = predict_proba(model, X_batch)
    meta = generate_fake_metadata(n_samples)

    alerts = mgr.process_predictions(preds, probs, model_name, metadata=meta)
    logger.info("Fired %d alerts for %d packets.", len(alerts), n_samples)
    
    for a in alerts[:5]:
        print(f"  [{a.severity:8s}] {a.source_ip} → {a.destination} "
              f"| conf={a.confidence:.0%} | {a.timestamp}")


def launch_dashboard() -> None:
    """Launch Streamlit dashboard."""
    logger.info("Launching Streamlit dashboard …")
    import subprocess
    
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard.py")
    
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run",
         dashboard_path,
         "--server.headless", "true",
         "--server.port", "8501"],
        check=True,
    )

# ─────────────────────────────────────────────────────────────
# Summary Table
# ─────────────────────────────────────────────────────────────

def _print_summary_table(results: dict) -> None:
    HEADER = (
        f"\n{'─'*70}\n"
        f"  {'Model':<20} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
        f"{'F1':>7} {'FPR':>7}\n"
        f"{'─'*70}"
    )
    print(HEADER)
    for name, m in results.items():
        print(
            f"  {name.replace('_', ' ').title():<20} "
            f"{m['accuracy']:>6.2f}% "
            f"{m['precision']:>6.2f}% "
            f"{m['recall']:>6.2f}% "
            f"{m['f1']:>6.2f}% "
            f"{m['false_positive_rate']:>6.2f}%"
        )
    print(f"{'─'*70}\n")


# ─────────────────────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Sentinel – Adaptive Cyber Threat Detection",
    )
    
    parser.add_argument(
        "--train", action="store_true",
        help="Train all models on the selected dataset.",
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate trained models and save charts.",
    )
    parser.add_argument(
        "--adaptive", action="store_true",
        help="Simulate adaptive retraining.",
    )
    parser.add_argument(
        "--alerts", action="store_true",
        help="Run alert-generation demo.",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Launch the Streamlit dashboard.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run train → evaluate → dashboard.",
    )
    parser.add_argument(
        "--dataset", default=config.ACTIVE_DATASET,
        choices=["nsl_kdd", "unsw", "live_dataset"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--model", default="random_forest",
        choices=list(MODEL_REGISTRY),
        help="Model for --adaptive.",
    )
    parser.add_argument(
        "--n-new", type=int, default=200,
        help="New samples for --adaptive.",
    )
    
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if not any(vars(args).values()):
        logger.info("No action. Use --help for options.")
        sys.exit(0)

    if args.train or args.all:
        run_training(args.dataset)

    if args.evaluate or args.all:
        run_evaluation(args.dataset)

    if args.adaptive:
        run_adaptive(args.model, args.dataset, args.n_new)

    if args.alerts:
        run_alert_demo(args.dataset)

    if args.dashboard or args.all:
        launch_dashboard()


if __name__ == "__main__":
    main()