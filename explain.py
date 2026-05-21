"""
xai/explain.py — Explainable AI engine
...
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

# Project root (explain.py lives next to prepare_data.py)
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from prepare_data import CLASS_NAMES

DATA_DIR = _ROOT / "data"
MODELS_DIR = _ROOT / "models"
LOGS_DIR = _ROOT / "logs"
XAI_OUTPUT_DIR = _ROOT / "logs" / "xai_outputs"
XAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_global_model():
    """Load the latest global XGBoost model."""
    model_path = MODELS_DIR / "global_model_latest.pkl"
    if not model_path.exists():
        checkpoints = sorted(MODELS_DIR.glob("global_model_round*.pkl"))
        if not checkpoints:
            raise FileNotFoundError(
                "No global model found. Run the simulation first:\n"
                "  python run_simulation.py"
            )
        model_path = checkpoints[-1]
    return joblib.load(model_path)


def load_test_data(hospital_id: str = "A") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load test split from a hospital partition for explanation."""
    data_path = DATA_DIR / f"hospital_{hospital_id}.csv"
    df = pd.read_csv(data_path)

    feature_file = DATA_DIR / "features.txt"
    with open(feature_file) as f:
        feature_cols = [l.strip() for l in f.readlines()]

    X = df[feature_cols].values.astype(np.float64)
    y = df["disease_class"].values.astype(int)

    scaler = joblib.load(DATA_DIR / "scaler.pkl")
    X = scaler.transform(X)

    # Take test split (last 20%)
    split = int(0.8 * len(X))
    return X[split:], y[split:], feature_cols


def compute_shap_values(model, X: np.ndarray, feature_names: List[str]):
    """Compute SHAP values using TreeExplainer (exact, fast for XGBoost)."""
    print("Computing SHAP values...")

    explainer = shap.TreeExplainer(model)

    sample_size = min(200, len(X))
    X_sample = X[:sample_size].astype(np.float32)

    shap_values = explainer.shap_values(X_sample)
    return explainer, shap_values, X_sample


def explain_single_patient(
    model,
    patient_features: np.ndarray,
    feature_names: List[str],
    explainer,
    shap_values_sample: np.ndarray,
    patient_idx: int = 0,
    true_label: Optional[int] = None,
) -> Dict:
    """
    Generate full explanation for a single patient:
    - Predicted class + probabilities
    - Top contributing features (from SHAP)
    - SHAP bar chart (saved to file)
    """
    # Prediction
    proba = model.predict_proba(patient_features.reshape(1, -1))[0]
    predicted_class = int(np.argmax(proba))
    predicted_label = CLASS_NAMES[predicted_class]
    confidence = float(proba[predicted_class])

    # SHAP values for this patient (for predicted class)
    patient_shap = shap_values_sample[predicted_class][patient_idx]

    # Top 8 features by absolute SHAP value
    top_indices = np.argsort(np.abs(patient_shap))[::-1][:8]
    top_features = []
    for idx in top_indices:
        top_features.append({
            "feature": feature_names[idx],
            "shap_value": float(patient_shap[idx]),
            "raw_value": float(patient_features[idx]),
            "direction": "increases risk" if patient_shap[idx] > 0 else "decreases risk"
        })

    # Generate SHAP bar chart
    chart_path = XAI_OUTPUT_DIR / f"shap_patient_{patient_idx}.png"
    _save_shap_chart(top_features, predicted_label, confidence, chart_path)

    explanation = {
        "patient_idx": patient_idx,
        "predicted_class": predicted_class,
        "predicted_label": predicted_label,
        "true_label": CLASS_NAMES.get(true_label, "Unknown") if true_label is not None else None,
        "confidence": round(confidence, 4),
        "all_class_probabilities": {
            CLASS_NAMES[i]: round(float(p), 4) for i, p in enumerate(proba)
        },
        "top_features": top_features,
        "chart_path": str(chart_path),
    }

    return explanation


def _save_shap_chart(top_features: List[Dict], predicted_label: str, confidence: float, path: Path):
    """Save a horizontal SHAP bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = [f["feature"] for f in top_features][::-1]
    values = [f["shap_value"] for f in top_features][::-1]
    colors = ["#E8593C" if v > 0 else "#3B8BD4" for v in values]

    bars = ax.barh(names, values, color=colors, edgecolor="none", height=0.6)
    ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")

    ax.set_xlabel("SHAP value (impact on prediction)", fontsize=10)
    ax.set_title(
        f"Prediction: {predicted_label} ({confidence*100:.1f}% confidence)\n"
        f"Feature contributions (red = increases risk, blue = decreases risk)",
        fontsize=10, pad=12
    )

    for bar, val in zip(bars, values):
        label = f" {val:+.3f}"
        ax.text(
            val + (0.002 if val >= 0 else -0.002),
            bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left" if val >= 0 else "right",
            fontsize=8, color="#333"
        )

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


def generate_llm_explanation(explanation: Dict) -> str:
    """
    Call Groq API (free tier) to generate a natural-language doctor-facing report.
    Falls back to a template-based report if Groq key is not set.
    """
    api_key = os.getenv("GROQ_API_KEY", "")

    if not api_key or api_key == "your_groq_api_key_here":
        return _template_explanation(explanation)

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        top_3 = explanation["top_features"][:3]
        feature_summary = "\n".join([
            f"  - {f['feature']}: SHAP={f['shap_value']:+.3f} ({f['direction']})"
            for f in top_3
        ])

        prompt = f"""You are a clinical AI assistant helping doctors understand a machine learning prediction.

Patient risk assessment result:
- Predicted condition: {explanation['predicted_label']}
- Model confidence: {explanation['confidence']*100:.1f}%
- Top contributing factors:
{feature_summary}

Write a concise clinical summary (3-4 sentences) that:
1. States the predicted risk clearly
2. Explains the main contributing factors in plain clinical language
3. Recommends what a clinician should verify or investigate
4. Notes that this is a decision support tool, not a diagnosis

Keep the tone professional and factual. Do not use bullet points."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"  Groq API call failed ({e}) — using template explanation")
        return _template_explanation(explanation)


def _template_explanation(explanation: Dict) -> str:
    """Fallback template-based explanation (no LLM needed)."""
    all_features = explanation["top_features"]
    label = explanation["predicted_label"]

    # Only features that actively drive this prediction (positive contribution)
    drivers = [f for f in all_features if f["shap_value"] > 0][:3]
    # Features that pull away from this prediction toward other conditions
    contra = [f for f in all_features if f["shap_value"] < 0][:2]

    if not drivers:
        driver_text = "No single feature strongly drives this prediction."
    else:
        driver_names = ", ".join(f["feature"].replace("_", " ") for f in drivers)
        driver_text = f"The primary factors supporting this prediction are: {driver_names}."

    if contra:
        contra_names = ", ".join(f["feature"].replace("_", " ") for f in contra)
        contra_text = (
            f" Note that {contra_names} show patterns more consistent with other conditions, "
            f"which may warrant broader clinical evaluation."
        )
    else:
        contra_text = ""

    return (
        f"The model predicts {label} with {explanation['confidence']*100:.1f}% confidence. "
        f"{driver_text}{contra_text} "
        f"Clinical validation is recommended. "
        f"This output is generated by a federated AI decision-support system and does not constitute a diagnosis."
    )


def run_explanation(patient_idx: int = 0, hospital_id: str = "A"):
    """Run full explanation pipeline for one patient."""
    print(f"\n[XAI] Loading global model...")
    model = load_global_model()

    print(f"[XAI] Loading test data from Hospital {hospital_id}...")
    X_test, y_test, feature_names = load_test_data(hospital_id)

    print(f"[XAI] Computing SHAP values ({min(200, len(X_test))} samples)...")
    explainer, shap_values, X_sample = compute_shap_values(model, X_test, feature_names)

    if patient_idx >= len(X_sample):
        patient_idx = 0
        print(f"  Patient index out of range, using patient 0")

    print(f"\n[XAI] Explaining patient {patient_idx}...")
    explanation = explain_single_patient(
        model=model,
        patient_features=X_sample[patient_idx],
        feature_names=feature_names,
        explainer=explainer,
        shap_values_sample=shap_values,
        patient_idx=patient_idx,
        true_label=int(y_test[patient_idx]) if patient_idx < len(y_test) else None,
    )

    print(f"\n  Predicted: {explanation['predicted_label']} ({explanation['confidence']*100:.1f}%)")
    if explanation["true_label"]:
        print(f"  True label: {explanation['true_label']}")
    print(f"\n  Top features:")
    for f in explanation["top_features"][:5]:
        print(f"    {f['feature']:<30} SHAP: {f['shap_value']:+.4f}  ({f['direction']})")

    print(f"\n[XAI] Generating LLM explanation...")
    llm_report = generate_llm_explanation(explanation)
    explanation["llm_report"] = llm_report

    print(f"\n--- CLINICAL REPORT ---")
    print(llm_report)
    print(f"------------------------")
    print(f"\n  SHAP chart saved → {explanation['chart_path']}")

    # Save full explanation to JSON
    output_path = XAI_OUTPUT_DIR / f"explanation_patient_{patient_idx}.json"
    with open(output_path, "w") as f:
        json.dump(explanation, f, indent=2)
    print(f"  Full explanation saved → {output_path}")

    return explanation


def main():
    parser = argparse.ArgumentParser(description="Generate XAI explanations")
    parser.add_argument("--patient-id", type=int, default=0,
                        help="Patient index to explain")
    parser.add_argument("--hospital", type=str, default="A",
                        choices=["A", "B", "C"],
                        help="Which hospital's test data to use")
    args = parser.parse_args()

    run_explanation(patient_idx=args.patient_id, hospital_id=args.hospital)


if __name__ == "__main__":
    main()
