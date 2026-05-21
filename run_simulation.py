"""
run_simulation.py — Federated XGBoost simulation (cyclic strategy).

Federated strategy: Cyclic training
  Each round, hospitals train sequentially. Hospital B starts from Hospital A's
  model; Hospital C starts from Hospital B's. The final hospital's model becomes
  the global model for the next round. This accumulates trees from all hospitals
  without sharing raw data.

Usage:
    python run_simulation.py
    python run_simulation.py --rounds 10
    python run_simulation.py --rounds 5 --verbose
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from fl_client import HospitalClient, _is_placeholder, _params_to_model

DATA_DIR   = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
LOGS_DIR   = Path(__file__).parent / "logs"

HOSPITALS = ["A", "B", "C"]


def check_data_ready() -> bool:
    return all((DATA_DIR / f"hospital_{h}.csv").exists() for h in HOSPITALS)


def run_simulation(num_rounds: int = 10):
    if not check_data_ready():
        print("""
╔══════════════════════════════════════════════════════════════╗
║  DATA NOT READY — run python3 prepare_data.py first          ║
╚══════════════════════════════════════════════════════════════╝
        """)
        return

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  FedXAI Simulation Starting                                  ║
║  Model:     XGBoost (cyclic federated training)              ║
║  Hospitals: {len(HOSPITALS)} (A, B, C — simulated locally)           ║
║  FL Rounds: {num_rounds:<49} ║
╚══════════════════════════════════════════════════════════════╝
    """)

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    clients = [HospitalClient(hospital_id=h, verbose=True) for h in HOSPITALS]

    # Start with placeholder (no global model yet)
    global_params = clients[0].get_parameters({})

    metrics_history: List[Dict] = []

    for rnd in range(1, num_rounds + 1):
        cfg = {"server_round": rnd}

        # ── Cyclic aggregation ─────────────────────────────────────
        # Each hospital trains from the previous hospital's model.
        # No raw data leaves the hospital; only the serialised model travels.
        for c in clients:
            global_params, _, _ = c.fit(global_params, cfg)

        # ── Evaluate global model on each hospital's test set ──────
        accs, f1s = [], []
        for c in clients:
            _loss, _n, ev = c.evaluate(global_params, cfg)
            accs.append(float(ev["accuracy"]))
            f1s.append(float(ev["f1_weighted"]))

        avg_acc = round(sum(accs) / len(accs), 4)
        avg_f1  = round(sum(f1s)  / len(f1s),  4)
        metrics_history.append({"round": rnd, "accuracy": avg_acc, "f1_weighted": avg_f1})

        # Save round checkpoint as pkl
        if not _is_placeholder(global_params):
            model = _params_to_model(global_params)
            joblib.dump(model, MODELS_DIR / f"global_model_round{rnd}.pkl")

    # Save final model
    if not _is_placeholder(global_params):
        model = _params_to_model(global_params)
        joblib.dump(model, MODELS_DIR / "global_model_latest.pkl")
        print(f"\n  Global model saved → {MODELS_DIR / 'global_model_latest.pkl'}")

    history_path = LOGS_DIR / "fl_training_history.json"
    with open(history_path, "w") as f:
        json.dump(metrics_history, f, indent=2)

    print(f"\n✓ Simulation complete! Results saved to {history_path}")
    print("\nTraining Summary:")
    print(f"  {'Round':<8} {'Accuracy':<12} {'F1':<12}")
    print(f"  {'-'*32}")
    for entry in metrics_history:
        print(f"  {entry['round']:<8} {entry['accuracy']:<12.4f} {entry['f1_weighted']:<12.4f}")

    print("\nNext steps:")
    print("  → Launch dashboard:  streamlit run app.py")


def main():
    parser = argparse.ArgumentParser(description="Run federated XGBoost simulation")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Number of FL rounds (default: 10)")
    args = parser.parse_args()
    run_simulation(num_rounds=args.rounds)


if __name__ == "__main__":
    main()
