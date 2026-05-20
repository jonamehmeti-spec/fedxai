"""
stage4_voi.py

Value of Information (VoI) based diagnostic test recommendation.

For each available test, computes the Expected Reduction in Diagnostic
Uncertainty (ERDU) — i.e., how much the test would reduce our uncertainty
about which disease the patient has.

The test with the highest ERDU is the "Next-Best-Test".
"""

import json
import math
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

KB_PATH = Path(__file__).parent / "knowledge_base.json"
with open(KB_PATH) as f:
    KB = json.load(f)


@dataclass
class TestRecommendation:
    test_id: str
    display_name: str
    voi_score: float
    diseases_resolved: List[str]
    sensitivity: float
    specificity: float
    invasiveness: str
    result_time_days: int
    normal_range: str
    abnormal_direction: str
    reasoning: str


def shannon_entropy(probabilities: List[float]) -> float:
    """Compute Shannon entropy of a probability distribution."""
    entropy = 0.0
    for p in probabilities:
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_posterior_if_positive(
    prior_probs: Dict[str, float],
    test_id: str,
    disease_tests: Dict[str, Dict]
) -> Dict[str, float]:
    """P(disease | test_positive) via Bayes."""
    updated = {}
    for disease_id, prior in prior_probs.items():
        tests = disease_tests.get(disease_id, {})
        sensitivity = tests.get(test_id, {}).get("sensitivity", 0.1)
        updated[disease_id] = prior * sensitivity

    total = sum(updated.values()) or 1e-9
    return {k: v / total for k, v in updated.items()}


def compute_posterior_if_negative(
    prior_probs: Dict[str, float],
    test_id: str,
    disease_tests: Dict[str, Dict]
) -> Dict[str, float]:
    """P(disease | test_negative) via Bayes."""
    updated = {}
    for disease_id, prior in prior_probs.items():
        tests = disease_tests.get(disease_id, {})
        sensitivity = tests.get(test_id, {}).get("sensitivity", 0.1)
        updated[disease_id] = prior * (1 - sensitivity)

    total = sum(updated.values()) or 1e-9
    return {k: v / total for k, v in updated.items()}


def compute_voi(
    differentials,
    test_id: str,
    test_data: Dict
) -> float:
    """
    Compute Expected Reduction in Diagnostic Uncertainty for a test.

    VoI = H(prior) - E[H(posterior)]
    where H is Shannon entropy and expectation is over test outcomes.
    """
    prior_probs = {d.disease_id: d.posterior_probability for d in differentials}
    prior_entropy = shannon_entropy(list(prior_probs.values()))

    # Build disease → test sensitivity map
    disease_tests = {}
    for d in differentials:
        disease_data = KB["rare_diseases"].get(d.disease_id, {})
        dist_tests = disease_data.get("distinguishing_tests", {})
        disease_tests[d.disease_id] = dist_tests

    # Compute P(test positive) marginalized over diseases
    sensitivity_avg = sum(
        disease_tests.get(d.disease_id, {}).get(test_id, {}).get("sensitivity", 0.05)
        * d.posterior_probability
        for d in differentials
    )
    p_positive = max(0.01, min(0.99, sensitivity_avg))
    p_negative = 1 - p_positive

    # Posterior entropy given positive result
    post_pos = compute_posterior_if_positive(prior_probs, test_id, disease_tests)
    h_pos = shannon_entropy(list(post_pos.values()))

    # Posterior entropy given negative result
    post_neg = compute_posterior_if_negative(prior_probs, test_id, disease_tests)
    h_neg = shannon_entropy(list(post_neg.values()))

    expected_posterior_entropy = p_positive * h_pos + p_negative * h_neg
    voi = prior_entropy - expected_posterior_entropy

    # Penalize slightly for invasiveness and time
    invasiveness_penalty = {"very_low": 0, "low": 0.02, "moderate": 0.1, "high": 0.2}
    penalty = invasiveness_penalty.get(test_data.get("invasiveness", "low"), 0.05)
    time_penalty = test_data.get("result_time_days", 1) * 0.01

    return max(0, round(voi - penalty - time_penalty, 4))


def get_diseases_resolved(test_id: str, differentials) -> List[str]:
    """Which diseases does this test most help clarify?"""
    resolved = []
    for d in differentials:
        disease_data = KB["rare_diseases"].get(d.disease_id, {})
        dist_tests = disease_data.get("distinguishing_tests", {})
        test_info = dist_tests.get(test_id, {})
        if test_info.get("sensitivity", 0) > 0.7 or test_info.get("specificity", 0) > 0.85:
            resolved.append(d.display_name)
    return resolved


def build_reasoning(test_data: Dict, diseases_resolved: List[str], voi: float) -> str:
    if not diseases_resolved:
        return f"This test provides general diagnostic information with a VoI score of {voi:.3f}."

    disease_str = " and ".join(diseases_resolved[:2])
    return (
        f"Ordering {test_data['display_name']} would most efficiently distinguish "
        f"{disease_str}. "
        f"With sensitivity {test_data['sensitivity']*100:.0f}% and specificity "
        f"{test_data['specificity']*100:.0f}%, this test offers the highest information "
        f"gain (VoI={voi:.3f}) given current diagnostic uncertainty."
    )


def recommend_tests(
    differentials,
    top_n: int = 4
) -> List[TestRecommendation]:
    """
    Main entry. Returns top-N tests ranked by Value of Information.

    Args:
        differentials: list of DiagnosisEntry from Stage 3
        top_n: how many tests to recommend

    Returns:
        List of TestRecommendation sorted by VoI descending
    """
    # Collect all unique tests across all differential diagnoses
    all_tests = {}
    for d in differentials:
        disease_data = KB["rare_diseases"].get(d.disease_id, {})
        for test_id, test_data in disease_data.get("distinguishing_tests", {}).items():
            if test_id not in all_tests:
                all_tests[test_id] = test_data

    recommendations = []
    for test_id, test_data in all_tests.items():
        voi = compute_voi(differentials, test_id, test_data)
        diseases_resolved = get_diseases_resolved(test_id, differentials)
        reasoning = build_reasoning(test_data, diseases_resolved, voi)

        recommendations.append(TestRecommendation(
            test_id=test_id,
            display_name=test_data["display_name"],
            voi_score=voi,
            diseases_resolved=diseases_resolved,
            sensitivity=test_data["sensitivity"],
            specificity=test_data["specificity"],
            invasiveness=test_data["invasiveness"],
            result_time_days=test_data["result_time_days"],
            normal_range=test_data.get("normal_range", "See lab reference"),
            abnormal_direction=test_data.get("abnormal_direction", "varies"),
            reasoning=reasoning
        ))

    recommendations.sort(key=lambda x: x.voi_score, reverse=True)
    return recommendations[:top_n]
