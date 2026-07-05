# ===== data_loader.py (FULLY CORRECTED) =====

import os
import logging
import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _ensure_dirs():
    """Create necessary directories if they don't exist."""
    for d in [
        config.DATA_DIR,
        config.MODEL_DIR,
        config.LOG_DIR,
        config.ALERT_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


def _validate_dataframe(df: pd.DataFrame, name: str) -> None:
    """Raise ValueError if the dataframe is invalid."""
    if df is None or df.empty:
        raise ValueError(f"[{name}] Dataset is empty or could not be loaded.")
    if df.shape[1] < 2:
        raise ValueError(f"[{name}] Dataset has fewer than 2 columns.")
    if df.shape[0] < 10:
        logger.warning(f"[{name}] Dataset has fewer than 10 rows.")
    
    logger.info("[%s] Loaded %d rows × %d columns", name, df.shape[0], df.shape[1])


def _validate_columns(df: pd.DataFrame, expected_cols: list, name: str) -> None:
    """Validate that expected columns exist in dataframe."""
    missing = set(expected_cols) - set(df.columns)
    if missing:
        logger.warning(f"[{name}] Missing columns: {missing}")
        # Add missing columns with default value
        for col in missing:
            df[col] = 0


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

LIVE_DATASET = os.path.join(config.DATA_DIR, "live_dataset.csv")


# ─────────────────────────────────────────────────────────────
# NSL-KDD Loader
# ─────────────────────────────────────────────────────────────

def load_nsl_kdd(
    train_path: str = config.NSL_KDD_TRAIN,
    test_path: str = config.NSL_KDD_TEST
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load NSL-KDD dataset."""
    _ensure_dirs()

    # Validate paths exist
    for path, name in [(train_path, "train"), (test_path, "test")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"NSL-KDD {name} file not found: {path}")

    # Load train data
    logger.info("Loading NSL-KDD train: %s", train_path)
    train_df = pd.read_csv(train_path, header=None, names=config.NSL_KDD_COLUMNS)
    _validate_dataframe(train_df, "NSL-KDD train")
    
    # Load test data
    logger.info("Loading NSL-KDD test: %s", test_path)
    test_df = pd.read_csv(test_path, header=None, names=config.NSL_KDD_COLUMNS)
    _validate_dataframe(test_df, "NSL-KDD test")

    # Convert labels to binary (0=normal, 1=attack)
    for df in (train_df, test_df):
        df[config.NSL_KDD_LABEL_COL] = (
            df[config.NSL_KDD_LABEL_COL]
            .str.strip()
            .str.lower() != config.NSL_KDD_NORMAL_LABEL.lower()
        ).astype(int)

    # Drop unnecessary columns
    drop_cols = [c for c in config.NSL_KDD_DROP_COLS if c in train_df.columns]
    train_df = train_df.drop(columns=drop_cols, errors="ignore")
    test_df = test_df.drop(columns=drop_cols, errors="ignore")

    logger.info("NSL-KDD ready. Train: %s | Test: %s", train_df.shape, test_df.shape)

    return train_df, test_df


# ─────────────────────────────────────────────────────────────
# UNSW-NB15 Loader
# ─────────────────────────────────────────────────────────────

def load_unsw(
    train_path: str = config.UNSW_TRAIN,
    test_path: str = config.UNSW_TEST
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load UNSW-NB15 dataset."""
    _ensure_dirs()

    # Validate paths exist
    for path, name in [(train_path, "train"), (test_path, "test")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"UNSW-NB15 {name} file not found: {path}")

    # Check for pyarrow
    try:
        import pyarrow  # noqa
    except ImportError:
        raise ImportError(
            "pyarrow is required for UNSW-NB15 dataset.\n"
            "Install: pip install pyarrow"
        )

    # Load train data
    logger.info("Loading UNSW-NB15 train: %s", train_path)
    train_df = pd.read_parquet(train_path)
    _validate_dataframe(train_df, "UNSW-NB15 train")

    # Load test data
    logger.info("Loading UNSW-NB15 test: %s", test_path)
    test_df = pd.read_parquet(test_path)
    _validate_dataframe(test_df, "UNSW-NB15 test")

    # Drop unnecessary columns
    for df in (train_df, test_df):
        for col in config.UNSW_DROP_COLS:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

    # Ensure label is integer
    for df in (train_df, test_df):
        df[config.UNSW_LABEL_COL] = df[config.UNSW_LABEL_COL].astype(int)

    logger.info("UNSW-NB15 ready. Train: %s | Test: %s", train_df.shape, test_df.shape)

    return train_df, test_df


# ─────────────────────────────────────────────────────────────
# LIVE Dataset Loader
# ─────────────────────────────────────────────────────────────

def load_live_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load lightweight live traffic dataset."""
    _ensure_dirs()

    if not os.path.isfile(LIVE_DATASET):
        raise FileNotFoundError(f"LIVE dataset file not found: {LIVE_DATASET}")

    logger.info("Loading LIVE dataset: %s", LIVE_DATASET)

    df = pd.read_csv(LIVE_DATASET)
    _validate_dataframe(df, "LIVE dataset")

    # Ensure label is integer
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)

    # Train/test split (80/20)
    train_df = df.sample(frac=0.8, random_state=config.RANDOM_SEED)
    test_df = df.drop(train_df.index)

    logger.info("LIVE dataset ready. Train: %s | Test: %s", train_df.shape, test_df.shape)

    return train_df, test_df


# ─────────────────────────────────────────────────────────────
# Unified Loader
# ─────────────────────────────────────────────────────────────

def load_dataset(dataset: str = config.ACTIVE_DATASET) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load selected dataset by name."""
    dataset = dataset.lower().strip()

    loaders = {
        "nsl_kdd": load_nsl_kdd,
        "unsw": load_unsw,
        "live_dataset": load_live_dataset,
    }

    if dataset not in loaders:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {list(loaders.keys())}"
        )

    return loaders[dataset]()


# ─────────────────────────────────────────────────────────────
# Label Column Helper
# ─────────────────────────────────────────────────────────────

def get_label_column(dataset: str = config.ACTIVE_DATASET) -> str:
    """Get the label column name for a given dataset."""
    dataset = dataset.lower().strip()
    
    label_cols = {
        "nsl_kdd": config.NSL_KDD_LABEL_COL,
        "unsw": config.UNSW_LABEL_COL,
        "live_dataset": "label",
    }
    
    if dataset not in label_cols:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    return label_cols[dataset]


# ─────────────────────────────────────────────────────────────
# Quick Test
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    dataset_name = config.ACTIVE_DATASET
    logger.info("Testing data_loader with dataset: %s", dataset_name)

    try:
        train_df, test_df = load_dataset(dataset_name)
        label_col = get_label_column(dataset_name)

        print("\n── Train Sample ──")
        print(train_df.head(3))

        print("\n── Label Distribution ──")
        print(train_df[label_col].value_counts())

        print("\n── Test Shape ──")
        print(test_df.shape)

    except Exception as exc:
        logger.error("Failed: %s", exc)
        raise