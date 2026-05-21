"""
Federated Learning Client — simulates a single hospital node.

Each hospital:
1. Loads its own local data partition (never shared)
2. Trains an XGBoost model on that data (or continues from global model)
3. Serialises the model and sends it to the FL server
4. Receives the updated global model and evaluates locally

Federated strategy: Cyclic training — hospitals train sequentially,
each starting from the previous hospital's model. This is the standard
approach for federated tree-based models (cannot average tree structures).
"""

import io
import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

import flwr as fl
from flwr.common import NDArrays, Scalar

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

CLASS_NAMES = {
    0: "Healthy",
    1: "Diabetes / Prediabetes",
    2: "High Cardiovascular Risk",
    3: "Cardiometabolic Risk",
}

# Trees added per hospital per round (cyclic FL accumulates across rounds)
TREES_PER_ROUND = 40


def load_hospital_data(hospital_id: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    data_path = DATA_DIR / f"hospital_{hospital_id}.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Hospital {hospital_id} data not found at {data_path}")

    df = pd.read_csv(data_path)
    feature_file = DATA_DIR / "features.txt"
    with open(feature_file) as f:
        feature_cols = [l.strip() for l in f.readlines()]

    X = df[feature_cols].values.astype(np.float32)
    y = df["disease_class"].values.astype(int)

    scaler_path = DATA_DIR / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X).astype(np.float32)

    return X, y, feature_cols


def _model_to_params(model: XGBClassifier) -> NDArrays:
    """Serialise XGBClassifier to bytes → numpy uint8 array for Flower."""
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return [np.frombuffer(buf.getvalue(), dtype=np.uint8)]


def _params_to_model(params: NDArrays) -> XGBClassifier:
    """Deserialise numpy uint8 array back to XGBClassifier."""
    buf = io.BytesIO(params[0].tobytes())
    return joblib.load(buf)


def _is_placeholder(params: NDArrays) -> bool:
    return len(params[0]) == 1 and params[0][0] == 0


def _make_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=TREES_PER_ROUND,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


class HospitalClient(fl.client.NumPyClient):
    def __init__(self, hospital_id: str, use_dp: bool = False,
                 dp_epsilon: float = 1.0, verbose: bool = True):
        self.hospital_id = hospital_id
        self.verbose = verbose

        print(f"\n[Hospital {hospital_id}] Loading data...")
        self.X, self.y, self.feature_names = load_hospital_data(hospital_id)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        n_classes = len(np.unique(self.y))
        print(f"[Hospital {hospital_id}] {len(self.X_train):,} train / "
              f"{len(self.X_test):,} test samples")
        print(f"[Hospital {hospital_id}] {n_classes} disease classes, "
              f"{len(self.feature_names)} features")

        self.model = _make_model()
        self._fitted = False

    def get_parameters(self, config: Dict) -> NDArrays:
        if not self._fitted:
            return [np.array([0], dtype=np.uint8)]  # placeholder
        return _model_to_params(self.model)

    def fit(self, parameters: NDArrays, config: Dict) -> Tuple[NDArrays, int, Dict]:
        round_num = config.get("server_round", 1)
        if self.verbose:
            print(f"\n[Hospital {self.hospital_id}] Round {round_num} — training locally...")

        if _is_placeholder(parameters):
            # First round — train from scratch
            self.model = _make_model()
            self.model.fit(self.X_train, self.y_train)
        else:
            # Continue training from the incoming global model (cyclic FL)
            global_model = _params_to_model(parameters)
            self.model = _make_model()
            self.model.fit(
                self.X_train, self.y_train,
                xgb_model=global_model.get_booster()
            )

        self._fitted = True

        train_preds = self.model.predict(self.X_train)
        train_acc = accuracy_score(self.y_train, train_preds)
        train_f1  = f1_score(self.y_train, train_preds, average="weighted", zero_division=0)

        if self.verbose:
            print(f"[Hospital {self.hospital_id}] Local accuracy: {train_acc:.4f} | F1: {train_f1:.4f}")

        model_path = MODELS_DIR / f"hospital_{self.hospital_id}_round{round_num}.pkl"
        joblib.dump(self.model, model_path)

        return (
            _model_to_params(self.model),
            len(self.X_train),
            {"train_accuracy": float(train_acc), "train_f1": float(train_f1)},
        )

    def evaluate(self, parameters: NDArrays, config: Dict) -> Tuple[float, int, Dict]:
        if _is_placeholder(parameters):
            return 1.0, len(self.X_test), {"accuracy": 0.0, "f1_weighted": 0.0}

        self.model = _params_to_model(parameters)
        self._fitted = True

        preds = self.model.predict(self.X_test)
        proba = self.model.predict_proba(self.X_test)
        acc  = accuracy_score(self.y_test, preds)
        f1   = f1_score(self.y_test, preds, average="weighted", zero_division=0)
        loss = log_loss(self.y_test, proba)

        if self.verbose:
            print(f"\n[Hospital {self.hospital_id}] EVALUATION — "
                  f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
            print(classification_report(
                self.y_test, preds,
                target_names=list(CLASS_NAMES.values()),
                zero_division=0
            ))

        return float(loss), len(self.X_test), {"accuracy": float(acc), "f1_weighted": float(f1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hospital", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--server", type=str, default="localhost:8080")
    args = parser.parse_args()

    client = HospitalClient(hospital_id=args.hospital)
    fl.client.start_numpy_client(server_address=args.server, client=client)


if __name__ == "__main__":
    main()
