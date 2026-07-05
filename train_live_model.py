# ===== train_live_model.py (FIXED) =====

"""
Trains the live-traffic Random Forest model and saves it for use by
LiveTrafficEngine.
"""

import os
import sys

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "live_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "live_rf_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "live_scaler.joblib")


def train_and_save(data_path: str = DATA_PATH) -> None:
    """Train and save model + scaler."""

    if not os.path.isfile(data_path):
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)

    print(f"Loading data from {data_path} …")
    df = pd.read_csv(data_path)

    if "label" not in df.columns:
        print("Error: 'label' column not found in data")
        sys.exit(1)

    # Split features/labels
    X = df.drop("label", axis=1)
    y = df["label"]

    # Train/test split (ONLY ONCE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale (fit ONLY on training data)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    print(f"Training on {len(X_train)} samples …")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate
    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2%}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"Model  → {MODEL_PATH}")
    print(f"Scaler → {SCALER_PATH}")
    print("Done!")


if __name__ == "__main__":
    train_and_save()