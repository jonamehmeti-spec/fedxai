"""
Federated Learning Client — simulates a single hospital node.

Each hospital:
1. Loads its own local data partition (never shared)
2. Trains a local model on that data
3. Optionally adds differential privacy noise to model parameters
4. Sends ONLY model weights to the FL server (never raw data)
5. Receives the updated global model and evaluates locally

Run via: python clients/fl_client.py --hospital A --server localhost:8080
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import joblib

import flwr as fl
from flwr.common import NDArrays, Scalar

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, log_loss
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Try to import diffprivlib (differential privacy)
try:
    from diffprivlib.models import LogisticRegression as DPLogisticRegression
    DP_AVAILABLE = True
except ImportError:
    DP_AVAILABLE = False
    print("  diffprivlib not installed — running without differential privacy")

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {
    0: "Healthy",
    1: "Diabetes / Prediabetes",
    2: "High Cardiovascular Risk",
    3: "Hypertension Risk"
}


def load_hospital_data(hospital_id: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and preprocess data for a specific hospital partition."""
    data_path = DATA_DIR / f"hospital_{hospital_id}.csv"

    if not data_path.exists():
        raise FileNotFoundError(
            f"Hospital {hospital_id} data not found at {data_path}\n"
            f"Please run: python data/prepare_data.py"
        )

    df = pd.read_csv(data_path)

    # Load feature names
    feature_file = DATA_DIR / "features.txt"
    if feature_file.exists():
        with open(feature_file) as f:
            feature_cols = [line.strip() for line in f.readlines()]
    else:
        feature_cols = [c for c in df.columns if c != "disease_class"]

    X = df[feature_cols].values.astype(np.float64)
    y = df["disease_class"].values.astype(int)

    # Standardize features
    scaler_path = DATA_DIR / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
    else:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    return X, y, feature_cols


def get_model_weights(model) -> NDArrays:
    """Extract model weights as a list of numpy arrays (for Flower)."""
    if hasattr(model, "coef_"):
        # LogisticRegression
        weights = [model.coef_.copy(), model.intercept_.copy()]
    else:
        raise ValueError("Unsupported model type for weight extraction")
    return weights


def set_model_weights(model, weights: NDArrays):
    """Set model weights from a list of numpy arrays."""
    if hasattr(model, "coef_"):
        model.coef_ = weights[0]
        model.intercept_ = weights[1]
    return model


class HospitalClient(fl.client.NumPyClient):
    """
    Flower client representing a single hospital.
    Trains locally, shares only model weights (+ optional DP noise).
    """

    def __init__(
        self,
        hospital_id: str,
        use_dp: bool = False,
        dp_epsilon: float = 1.0,
        verbose: bool = True
    ):
        self.hospital_id = hospital_id
        self.use_dp = use_dp and DP_AVAILABLE
        self.dp_epsilon = dp_epsilon
        self.verbose = verbose

        print(f"\n[Hospital {hospital_id}] Loading data...")
        self.X, self.y, self.feature_names = load_hospital_data(hospital_id)

        # 80/20 stratified train-test split (shuffle to avoid class ordering bias)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        n_classes = len(np.unique(self.y))
        print(f"[Hospital {hospital_id}] {len(self.X_train):,} train / {len(self.X_test):,} test samples")
        print(f"[Hospital {hospital_id}] {n_classes} disease classes, {len(self.feature_names)} features")

        # Initialize model
        if self.use_dp:
            self.model = DPLogisticRegression(
                epsilon=dp_epsilon,
                max_iter=100,
                solver="lbfgs",
                multi_class="multinomial"
            )
            print(f"[Hospital {hospital_id}] Using Differential Privacy (ε={dp_epsilon})")
        else:
            self.model = LogisticRegression(
                max_iter=500,
                solver="lbfgs",
                multi_class="multinomial",
                C=1.0,
                class_weight="balanced",
                random_state=42
            )

    def get_parameters(self, config: Dict) -> NDArrays:
        """Called by Flower to get current model parameters."""
        try:
            return get_model_weights(self.model)
        except Exception:
            # Model not yet fitted — return zeros
            n_features = self.X_train.shape[1]
            n_classes = len(np.unique(self.y_train))
            return [
                np.zeros((n_classes, n_features)),
                np.zeros(n_classes)
            ]

    def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        """
        Called by FL server to train locally.
        Sets global parameters → trains locally → returns updated weights.
        """
        round_num = config.get("server_round", 1)

        # Load global model weights from server
        try:
            self.model = set_model_weights(self.model, parameters)
            # Warm-start: signal model is partially fitted
            self.model.warm_start = True
        except Exception:
            pass

        # LOCAL TRAINING — raw data never leaves this function
        if self.verbose:
            print(f"\n[Hospital {self.hospital_id}] Round {round_num} — training locally...")

        self.model.fit(self.X_train, self.y_train)

        train_preds = self.model.predict(self.X_train)
        train_acc = accuracy_score(self.y_train, train_preds)
        train_f1 = f1_score(self.y_train, train_preds, average="weighted", zero_division=0)

        if self.verbose:
            print(f"[Hospital {self.hospital_id}] Local accuracy: {train_acc:.4f} | F1: {train_f1:.4f}")

        # Save local model checkpoint
        model_path = MODELS_DIR / f"hospital_{self.hospital_id}_round{round_num}.pkl"
        joblib.dump(self.model, model_path)

        return (
            get_model_weights(self.model),
            len(self.X_train),
            {"train_accuracy": float(train_acc), "train_f1": float(train_f1)}
        )

    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        """
        Called by FL server to evaluate the global model on local test data.
        """
        self.model = set_model_weights(self.model, parameters)

        preds = self.model.predict(self.X_test)
        proba = self.model.predict_proba(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        f1 = f1_score(self.y_test, preds, average="weighted", zero_division=0)
        loss = log_loss(self.y_test, proba)

        if self.verbose:
            print(f"\n[Hospital {self.hospital_id}] EVALUATION — Accuracy: {acc:.4f} | F1: {f1:.4f}")
            print(classification_report(
                self.y_test, preds,
                target_names=list(CLASS_NAMES.values()),
                zero_division=0
            ))

        return (
            float(loss),
            len(self.X_test),
            {"accuracy": float(acc), "f1_weighted": float(f1)}
        )


def main():
    parser = argparse.ArgumentParser(description="Start a hospital FL client")
    parser.add_argument("--hospital", type=str, default="A", choices=["A", "B", "C"],
                        help="Hospital identifier (A, B, or C)")
    parser.add_argument("--server", type=str, default="localhost:8080",
                        help="FL server address")
    parser.add_argument("--dp", action="store_true",
                        help="Enable differential privacy")
    parser.add_argument("--dp-epsilon", type=float, default=1.0,
                        help="Differential privacy epsilon (lower = more private)")
    args = parser.parse_args()

    client = HospitalClient(
        hospital_id=args.hospital,
        use_dp=args.dp,
        dp_epsilon=args.dp_epsilon
    )

    print(f"\n[Hospital {args.hospital}] Connecting to FL server at {args.server}...")
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client
    )


if __name__ == "__main__":
    main()
