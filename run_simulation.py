"""
run_simulation.py — Run the full federated learning simulation in a single process.

Runs FedAvg across three hospital clients in one process (no Ray required).
Flower's `start_simulation` needs Ray; this script uses the same client code with
a local weighted FedAvg loop so `pip install flwr[simulation]` is optional.

Usage:
    python run_simulation.py
    python run_simulation.py --rounds 10 --dp  (with differential privacy)
    python run_simulation.py --rounds 3 --no-dp --verbose

After running, check:
  - logs/fl_training_history.json  → per-round metrics
  - models/global_model_latest.npy → final global model weights
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from fl_client import HospitalClient
from flwr.common import NDArrays

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
LOGS_DIR = Path(__file__).parent / "logs"

HOSPITALS = ["A", "B", "C"]


def _save_weight_tuple(path: Path, weights: NDArrays) -> None:
    """Persist (coef, intercept) as .npy readable by np.load(..., allow_pickle=True)."""
    arr = np.empty(len(weights), dtype=object)
    for i, w in enumerate(weights):
        arr[i] = w
    np.save(str(path), arr)


def check_data_ready() -> bool:
    for h in HOSPITALS:
        if not (DATA_DIR / f"hospital_{h}.csv").exists():
            return False
    return True


def _fedavg(weights_list: List[NDArrays], num_examples: List[int]) -> NDArrays:
    """Sample-weighted average of client weight tuples (coef, intercept)."""
    total = float(sum(num_examples))
    return [
        sum(w[0] * n for w, n in zip(weights_list, num_examples)) / total,
        sum(w[1] * n for w, n in zip(weights_list, num_examples)) / total,
    ]


def run_simulation(
    num_rounds: int = 5,
    use_dp: bool = False,
    dp_epsilon: float = 1.0,
):
    if not check_data_ready():
        print("""
╔══════════════════════════════════════════════════════════════╗
║  DATA NOT READY                                              ║
║                                                              ║
║  Please prepare the dataset first:                           ║
║  1. Download the CDC BRFSS dataset from Kaggle               ║
║     https://www.kaggle.com/datasets/alexteboul/              ║
║     diabetes-health-indicators-dataset                       ║
║  2. Save as: data/raw_data.csv                               ║
║  3. Run: python3 prepare_data.py                             ║
║                                                              ║
║  Then re-run: python run_simulation.py                       ║
╚══════════════════════════════════════════════════════════════╝
        """)
        return

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  FedXAI Simulation Starting                                  ║
║  Hospitals: {len(HOSPITALS)} (A, B, C — simulated locally)           ║
║  FL Rounds: {num_rounds:<49} ║
║  Differential Privacy: {'ON  (ε=' + str(dp_epsilon) + ')' if use_dp else 'OFF':<38} ║
╚══════════════════════════════════════════════════════════════╝
    """)

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    clients = [
        HospitalClient(
            hospital_id=h,
            use_dp=use_dp,
            dp_epsilon=dp_epsilon,
            verbose=True,
        )
        for h in HOSPITALS
    ]
    global_params = clients[0].get_parameters({})

    metrics_history: List[Dict] = []
    for rnd in range(1, num_rounds + 1):
        cfg = {"server_round": rnd}
        weights_list: List[NDArrays] = []
        nums: List[int] = []
        for c in clients:
            w, n, _ = c.fit(global_params, cfg)
            weights_list.append(w)
            nums.append(n)
        global_params = _fedavg(weights_list, nums)

        accs: List[float] = []
        f1s: List[float] = []
        for c in clients:
            _loss, _n, ev = c.evaluate(global_params, cfg)
            accs.append(float(ev["accuracy"]))
            f1s.append(float(ev["f1_weighted"]))

        metrics_history.append({
            "round": rnd,
            "accuracy": round(sum(accs) / len(accs), 4),
            "f1_weighted": round(sum(f1s) / len(f1s), 4),
        })

        _save_weight_tuple(MODELS_DIR / f"global_model_round{rnd}.npy", global_params)

    _save_weight_tuple(MODELS_DIR / "global_model_latest.npy", global_params)

    history_path = LOGS_DIR / "fl_training_history.json"
    with open(history_path, "w") as f:
        json.dump(metrics_history, f, indent=2)

    print(f"\n✓ Simulation complete! Results saved to {history_path}")
    print("\nTraining Summary:")
    print(f"  {'Round':<8} {'Accuracy':<12} {'F1':<12}")
    print(f"  {'-'*32}")

    for entry in metrics_history:
        r = entry.get("round", "?")
        acc = entry.get("accuracy", entry.get("avg_accuracy", "—"))
        f1 = entry.get("f1_weighted", entry.get("avg_f1_weighted", "—"))
        acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
        f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
        print(f"  {r:<8} {acc_str:<12} {f1_str:<12}")

    print("\nNext steps:")
    print("  → Run XAI analysis:       python xai/explain.py")
    print("  → Launch dashboard:       streamlit run dashboard/app.py")


def main():
    parser = argparse.ArgumentParser(description="Run federated learning simulation")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of FL rounds (default: 5)")
    parser.add_argument("--dp", action="store_true", default=False,
                        help="Enable differential privacy")
    parser.add_argument("--dp-epsilon", type=float, default=1.0,
                        help="Privacy budget (lower = more private, default: 1.0)")
    args = parser.parse_args()

    run_simulation(
        num_rounds=args.rounds,
        use_dp=args.dp,
        dp_epsilon=args.dp_epsilon,
    )


if __name__ == "__main__":
    main()
