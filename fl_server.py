"""
Federated Learning Server — aggregates model weights from hospital clients.

Uses the FedAvg (Federated Averaging) algorithm:
  global_weights = weighted average of all client weights
  (weighted by number of training samples per client)

Run: python server/fl_server.py

Then in separate terminals, run each hospital client:
  python clients/fl_client.py --hospital A
  python clients/fl_client.py --hospital B
  python clients/fl_client.py --hospital C

Or use the simulation script: python run_simulation.py
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

LOGS_DIR = Path(__file__).parent.parent / "logs"
MODELS_DIR = Path(__file__).parent.parent / "models"
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


class FedXAIStrategy(FedAvg):
    """
    Custom FedAvg strategy that:
    - Logs metrics per round to JSON (for the dashboard to read)
    - Saves the global model weights after each round
    - Prints a clean summary of federated training progress
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_metrics: List[Dict] = []
        self.best_accuracy = 0.0
        self.history_path = LOGS_DIR / "fl_training_history.json"

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using FedAvg, then save global model."""

        if not results:
            return None, {}

        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_params is not None:
            # Save global model weights
            weights = parameters_to_ndarrays(aggregated_params)
            weights_path = MODELS_DIR / f"global_model_round{server_round}.npy"
            np.save(str(weights_path), np.array(weights, dtype=object))
            # Always overwrite the "latest" global model
            np.save(str(MODELS_DIR / "global_model_latest.npy"), np.array(weights, dtype=object))

        # Collect per-client metrics
        client_metrics = []
        for client, fit_res in results:
            m = fit_res.metrics or {}
            client_metrics.append({
                "num_examples": fit_res.num_examples,
                "train_accuracy": m.get("train_accuracy", 0),
                "train_f1": m.get("train_f1", 0),
            })

        # Weighted average of training metrics
        total_examples = sum(cm["num_examples"] for cm in client_metrics)
        avg_train_acc = sum(
            cm["train_accuracy"] * cm["num_examples"] for cm in client_metrics
        ) / total_examples if total_examples > 0 else 0

        print(f"\n{'='*60}")
        print(f"  ROUND {server_round} — Aggregation complete")
        print(f"  Clients: {len(results)} | Total samples: {total_examples:,}")
        print(f"  Avg train accuracy: {avg_train_acc:.4f}")
        print(f"{'='*60}")

        return aggregated_params, {"avg_train_accuracy": avg_train_acc}

    def aggregate_evaluate(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation metrics and log to file."""

        if not results:
            return None, {}

        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Collect per-client eval metrics
        total_examples = sum(r.num_examples for _, r in results)
        avg_acc = sum(
            (r.metrics.get("accuracy", 0) * r.num_examples) for _, r in results
        ) / total_examples if total_examples > 0 else 0

        avg_f1 = sum(
            (r.metrics.get("f1_weighted", 0) * r.num_examples) for _, r in results
        ) / total_examples if total_examples > 0 else 0

        round_data = {
            "round": server_round,
            "avg_accuracy": round(avg_acc, 4),
            "avg_f1_weighted": round(avg_f1, 4),
            "avg_loss": round(float(loss) if loss else 0, 4),
            "num_clients": len(results),
            "total_eval_samples": total_examples,
            "client_metrics": [
                {
                    "num_examples": r.num_examples,
                    "accuracy": r.metrics.get("accuracy", 0),
                    "f1_weighted": r.metrics.get("f1_weighted", 0),
                }
                for _, r in results
            ]
        }

        self.round_metrics.append(round_data)

        # Persist metrics history
        with open(self.history_path, "w") as f:
            json.dump(self.round_metrics, f, indent=2)

        if avg_acc > self.best_accuracy:
            self.best_accuracy = avg_acc
            # Save best model separately
            latest = MODELS_DIR / "global_model_latest.npy"
            if latest.exists():
                import shutil
                shutil.copy(latest, MODELS_DIR / "global_model_best.npy")

        print(f"\n  EVAL ROUND {server_round}:")
        print(f"  Accuracy: {avg_acc:.4f} | F1: {avg_f1:.4f} | Best so far: {self.best_accuracy:.4f}")

        return loss, {"avg_accuracy": avg_acc, "avg_f1_weighted": avg_f1}


def start_server(
    num_rounds: int = 5,
    min_clients: int = 3,
    server_address: str = "0.0.0.0:8080"
):
    strategy = FedXAIStrategy(
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        on_fit_config_fn=lambda round_num: {"server_round": round_num},
        on_evaluate_config_fn=lambda round_num: {"server_round": round_num},
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  FedXAI — Federated Learning Server                          ║
║  Address: {server_address:<49} ║
║  Rounds:  {num_rounds:<49} ║
║  Min clients: {min_clients:<44} ║
║                                                              ║
║  Waiting for {min_clients} hospital clients to connect...           ║
╚══════════════════════════════════════════════════════════════╝
    """)

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


def main():
    parser = argparse.ArgumentParser(description="Start the FL aggregation server")
    parser.add_argument("--rounds", type=int, default=5,
                        help="Number of federated learning rounds")
    parser.add_argument("--min-clients", type=int, default=3,
                        help="Minimum number of hospital clients required")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080",
                        help="Server address")
    args = parser.parse_args()

    start_server(
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        server_address=args.address
    )


if __name__ == "__main__":
    main()
