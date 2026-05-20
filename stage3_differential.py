"""
stage3_differential.py

Bayesian differential diagnosis ranking.
Takes Stage 2 mimic alerts + patient features and computes
posterior probability for each diagnosis using Bayes' theorem.

P(disease | features) ∝ P(features | disease) × P(disease)
"""

import json
import math
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, field

KB_PATH = Path(__file__).parent / "knowledge_base.json"
with open(KB_PATH) as f:
    KB = json.load(f)


@dataclass
class DiagnosisEntry:
    disease_id: str
    display_name: str
    category: str
    posterior_probability: float
    prior_probability: float
    likelihood_score: float
    confidence_interval: tuple
    supporting_features: List[str]
    against_features: List[str]
    uncertainty: str


def compute_likelihood(disease_data: Dict, active_features: List[str]) -> float:
    clinical_features = disease_data.get("clinical_features", {})
    if not clinical_features:
        return 0.01

    log_likelihood = 0.0
    for feature, meta in clinical_features.items():
        p_feature_given_disease = meta["present_in_pct"]
        p_feature_given_no_disease = 0.15

        if feature in active_features:
            if p_feature_given_disease > 0:
                log_likelihood += math.log(
                    p_feature_given_disease / p_feature_given_no_disease
                )
        else:
            p_absent_given_disease = 1 - p_feature_given_disease
            p_absent_given_no_disease = 1 - p_feature_given_no_disease
            if p_absent_given_disease > 0 and p_absent_given_no_disease > 0:
                log_likelihood += math.log(
                    p_absent_given_disease / p_absent_given_no_disease
                ) * 0.3

    # Clip to avoid extreme values
    likelihood = math.exp(max(-6, min(6, log_likelihood)))
    return likelihood


def get_prior(disease_id: str, common_flags: List[str]) -> float:
    disease_data = KB["rare_diseases"].get(disease_id, {})
    if "essential_hypertension" in common_flags:
        prior = disease_data.get("prevalence_in_hypertension",
                disease_data.get("prevalence_general", 0.001))
    elif "type2_diabetes" in common_flags:
        prior = disease_data.get("prevalence_in_diabetes",
                disease_data.get("prevalence_general", 0.001))
    else:
        prior = disease_data.get("prevalence_general", 0.001)
    return prior


def classify_uncertainty(posterior: float, n_features: int) -> str:
    if n_features >= 4 and posterior > 0.3:
        return "low"
    elif n_features >= 2 or posterior > 0.15:
        return "moderate"
    else:
        return "high"


def rank_differentials(
    mimic_alerts,
    patient_values: Dict,
    common_flags: List[str],
    active_features: List[str],
    top_n: int = 6
) -> List[DiagnosisEntry]:

    raw_results = []

    for alert in mimic_alerts:
        disease_id = alert.disease_id
        disease_data = KB["rare_diseases"].get(disease_id, {})

        prior = get_prior(disease_id, common_flags)
        likelihood = compute_likelihood(disease_data, active_features)
        unnormalized = prior * likelihood

        clinical_features = disease_data.get("clinical_features", {})
        supporting = [f for f in clinical_features if f in active_features]
        against = [
            f for f in clinical_features
            if f not in active_features
            and clinical_features[f]["present_in_pct"] > 0.7
        ]

        raw_results.append({
            "disease_id": disease_id,
            "display_name": alert.display_name,
            "category": alert.category,
            "unnormalized": unnormalized,
            "prior": prior,
            "likelihood": likelihood,
            "supporting": supporting,
            "against": against[:3],
        })

    # Normalize so all posteriors sum to 1
    total = sum(r["unnormalized"] for r in raw_results)
    if total == 0:
        total = 1e-9

    entries = []
    for r in raw_results:
        posterior = r["unnormalized"] / total

        # CI: simple ±15% relative uncertainty, clipped to [0, 1]
        margin = posterior * 0.3
        ci_lower = round(max(0.0, posterior - margin), 3)
        ci_upper = round(min(1.0, posterior + margin), 3)

        uncertainty = classify_uncertainty(posterior, len(r["supporting"]))

        entries.append(DiagnosisEntry(
            disease_id=r["disease_id"],
            display_name=r["display_name"],
            category=r["category"],
            posterior_probability=round(posterior, 4),
            prior_probability=r["prior"],
            likelihood_score=r["likelihood"],
            confidence_interval=(ci_lower, ci_upper),
            supporting_features=r["supporting"],
            against_features=r["against"],
            uncertainty=uncertainty
        ))

    entries.sort(key=lambda x: x.posterior_probability, reverse=True)
    return entries[:top_n]