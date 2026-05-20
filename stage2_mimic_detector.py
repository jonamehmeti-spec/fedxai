"""
stage2_mimic_detector.py

Takes Stage 1 output (common disease probabilities + patient indicators)
and flags which rare diseases could be mimicking the common presentation.

Returns a list of MimicAlert objects with confidence scores.
"""

import json
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

KB_PATH = Path(__file__).parent / "knowledge_base.json"

with open(KB_PATH) as f:
    KB = json.load(f)


@dataclass
class MimicAlert:
    disease_id: str
    display_name: str
    category: str
    confidence: float
    matched_features: List[str]
    matched_sources: List[str]  # human-readable patient inputs that drove the score
    avg_diagnostic_delay_months: int
    red_flags: List[str]
    description: str


def get_common_flags(stage1_proba: Dict[str, float], threshold: float = 0.25) -> List[str]:
    """Map Stage 1 class probabilities to common disease flags."""
    flag_map = {
        "Hypertension Risk": "essential_hypertension",
        "Diabetes / Prediabetes": "type2_diabetes",
        "High Cardiovascular Risk": "high_cvd_risk",
        "Healthy": None
    }
    flags = []
    for label, prob in stage1_proba.items():
        if prob >= threshold and flag_map.get(label):
            flags.append(flag_map[label])
    return flags


def map_patient_features(patient_values: Dict) -> List[str]:
    """Map raw patient input values to clinical feature names."""
    active_features = []

    if patient_values.get("HighBP", 0) == 1:
        active_features.extend(["hypertension", "hypertension_resistant"])
    if patient_values.get("HighChol", 0) == 1:
        active_features.append("elevated_cholesterol")
    if patient_values.get("BMI", 25) > 30:
        active_features.extend(["weight_gain", "weight_gain_central"])
    if patient_values.get("HeartDiseaseorAttack", 0) == 1:
        active_features.extend(["recurrent_thrombosis", "cardiac_valve_disease"])
    if patient_values.get("Stroke", 0) == 1:
        active_features.extend(["early_stroke", "recurrent_thrombosis"])
    if patient_values.get("PhysActivity", 1) == 0:
        active_features.extend(["fatigue", "muscle_weakness"])
    if patient_values.get("GenHlth", 3) >= 4:
        active_features.extend(["fatigue", "fatigue_chronic"])
    if patient_values.get("DiffWalk", 0) == 1:
        active_features.extend(["muscle_weakness", "joint_pain"])

    # ── Female-specific mappings ───────────────────────────────────
    if patient_values.get("GestationalDiabetes", 0) == 1:
        # Strong signal for metabolic/insulin resistance conditions
        active_features.extend(["hyperglycemia", "weight_gain_central"])
    if patient_values.get("PCOS", 0) == 1:
        active_features.extend(["hyperglycemia", "weight_gain", "hypertension"])
    if patient_values.get("IrregularPeriods", 0) == 1:
        active_features.extend(["weight_gain", "fatigue"])
    if patient_values.get("Preeclampsia", 0) == 1:
        active_features.extend(["hypertension", "hypertension_resistant", "recurrent_thrombosis"])
    if patient_values.get("MenopauseStatus", "not_applicable") == "post_menopausal":
        active_features.extend(["elevated_cholesterol", "hypertension_diastolic"])
    if patient_values.get("HormonalTherapy", 0) == 1:
        active_features.extend(["hypertension", "recurrent_thrombosis"])

    # ── Male-specific mappings ─────────────────────────────────────
    if patient_values.get("ErectileDysfunction", 0) == 1:
        # ED is a recognised early marker of atherosclerosis
        active_features.extend(["hypertension", "elevated_cholesterol", "cardiac_valve_disease"])
    if patient_values.get("ProstatIssues", 0) == 1:
        active_features.extend(["fatigue", "weight_gain"])
    if patient_values.get("TestosteroneTherapy", 0) == 1:
        active_features.extend(["elevated_cholesterol", "hypertension", "recurrent_thrombosis"])

    return list(set(active_features))


def get_source_map(patient_values: Dict) -> Dict[str, List[str]]:
    """Returns {clinical_feature: [readable patient input labels]} for traceability."""
    sources: Dict[str, List[str]] = {}

    def add(feature: str, label: str):
        sources.setdefault(feature, [])
        if label not in sources[feature]:
            sources[feature].append(label)

    if patient_values.get("HighBP", 0) == 1:
        add("hypertension", "High Blood Pressure")
        add("hypertension_resistant", "High Blood Pressure")
    if patient_values.get("HighChol", 0) == 1:
        add("elevated_cholesterol", "High Cholesterol")
    if patient_values.get("BMI", 25) > 30:
        add("weight_gain", "BMI > 30")
        add("weight_gain_central", "BMI > 30")
    if patient_values.get("HeartDiseaseorAttack", 0) == 1:
        add("recurrent_thrombosis", "Heart Disease History")
        add("cardiac_valve_disease", "Heart Disease History")
    if patient_values.get("Stroke", 0) == 1:
        add("early_stroke", "Stroke History")
        add("recurrent_thrombosis", "Stroke History")
    if patient_values.get("PhysActivity", 1) == 0:
        add("fatigue", "Physical Inactivity")
        add("muscle_weakness", "Physical Inactivity")
    if patient_values.get("GenHlth", 3) >= 4:
        add("fatigue", "Poor General Health")
        add("fatigue_chronic", "Poor General Health")
    if patient_values.get("DiffWalk", 0) == 1:
        add("muscle_weakness", "Difficulty Walking")
        add("joint_pain", "Difficulty Walking")
    # Female-specific
    if patient_values.get("GestationalDiabetes", 0) == 1:
        add("hyperglycemia", "Gestational Diabetes")
        add("weight_gain_central", "Gestational Diabetes")
    if patient_values.get("PCOS", 0) == 1:
        add("hyperglycemia", "PCOS")
        add("weight_gain", "PCOS")
        add("hypertension", "PCOS")
    if patient_values.get("IrregularPeriods", 0) == 1:
        add("weight_gain", "Irregular Periods")
        add("fatigue", "Irregular Periods")
    if patient_values.get("Preeclampsia", 0) == 1:
        add("hypertension", "Preeclampsia History")
        add("hypertension_resistant", "Preeclampsia History")
        add("recurrent_thrombosis", "Preeclampsia History")
    if patient_values.get("MenopauseStatus") == "post_menopausal":
        add("elevated_cholesterol", "Post-Menopausal Status")
        add("hypertension_diastolic", "Post-Menopausal Status")
    if patient_values.get("HormonalTherapy", 0) == 1:
        add("hypertension", "Hormonal Therapy")
        add("recurrent_thrombosis", "Hormonal Therapy")
    # Male-specific
    if patient_values.get("ErectileDysfunction", 0) == 1:
        add("hypertension", "Erectile Dysfunction")
        add("elevated_cholesterol", "Erectile Dysfunction")
        add("cardiac_valve_disease", "Erectile Dysfunction")
    if patient_values.get("ProstatIssues", 0) == 1:
        add("fatigue", "Prostate Issues")
        add("weight_gain", "Prostate Issues")
    if patient_values.get("TestosteroneTherapy", 0) == 1:
        add("elevated_cholesterol", "Testosterone Therapy")
        add("hypertension", "Testosterone Therapy")
        add("recurrent_thrombosis", "Testosterone Therapy")

    return sources


FEMALE_ONLY_FEATURES = {"vaginal_dryness", "recurrent_miscarriage", "menstrual_irregularity"}
MALE_ONLY_FEATURES = {"erectile_dysfunction", "prostate_enlargement"}

FEMALE_ONLY_KEYWORDS = {"woman", "women", "pregnancy", "pregnant", "miscarriage", "menstrual", "ovarian", "uterine"}
MALE_ONLY_KEYWORDS = {"prostate", "testicular"}


def _filter_red_flags(red_flags: List[str], patient_sex: str) -> List[str]:
    """Remove red flags that mention gender-specific anatomy irrelevant to this patient."""
    if patient_sex == "other":
        return red_flags
    exclude = FEMALE_ONLY_KEYWORDS if patient_sex == "male" else MALE_ONLY_KEYWORDS
    return [
        flag for flag in red_flags
        if not any(kw in flag.lower() for kw in exclude)
    ]


def filter_features_by_sex(clinical_features: Dict, patient_sex: str) -> Dict:
    """Remove clinical features that are irrelevant to the patient's sex."""
    if patient_sex == "female":
        return {k: v for k, v in clinical_features.items() if k not in MALE_ONLY_FEATURES}
    if patient_sex == "male":
        return {k: v for k, v in clinical_features.items() if k not in FEMALE_ONLY_FEATURES}
    return clinical_features  # "other" — no filtering


def compute_mimic_confidence(
    disease_data: Dict,
    active_features: List[str],
    common_flags: List[str],
    patient_sex: str = "other",
) -> tuple:
    """
    Compute how likely a rare disease is given active features.
    Returns (confidence_score, matched_features).
    """
    clinical_features = filter_features_by_sex(
        disease_data.get("clinical_features", {}), patient_sex
    )
    matched = []
    total_weight = 0
    matched_weight = 0

    for feature, meta in clinical_features.items():
        weight = meta["weight"]
        total_weight += weight
        if feature in active_features:
            matched_weight += weight
            matched.append(feature)

    if total_weight == 0:
        return 0.0, []

    base_score = matched_weight / total_weight

    # Boost if required common flags are present
    required_flags = disease_data.get("trigger_indicators", {}).get("required_common_flags", [])
    if required_flags:
        flags_matched = sum(1 for f in required_flags if f in common_flags)
        flag_boost = flags_matched / len(required_flags) * 0.3
        base_score = min(1.0, base_score + flag_boost)

    # Prevalence prior (log scale to avoid extreme suppression)
    prevalence = disease_data.get("prevalence_general", 0.001)
    import math
    prevalence_factor = 1 + math.log10(max(prevalence, 0.0001) * 1000) * 0.1
    final_score = min(1.0, base_score * prevalence_factor)

    return round(final_score, 3), matched


def detect_mimics(
    stage1_proba: Dict[str, float],
    patient_values: Dict,
    min_confidence: float = 0.05,
    patient_sex: str = "other",
) -> List[MimicAlert]:
    """
    Main entry point. Returns ranked list of rare disease alerts.

    Args:
        stage1_proba: {"Healthy": 0.1, "Diabetes / Prediabetes": 0.6, ...}
        patient_values: raw patient input dict
        min_confidence: minimum score to include in results

    Returns:
        List of MimicAlert sorted by confidence descending
    """
    common_flags = get_common_flags(stage1_proba)
    active_features = map_patient_features(patient_values)
    source_map = get_source_map(patient_values)

    alerts = []
    for disease_id, disease_data in KB["rare_diseases"].items():
        confidence, matched = compute_mimic_confidence(
            disease_data, active_features, common_flags, patient_sex
        )

        if confidence >= min_confidence:
            # Collect unique readable sources for the matched features
            sources = []
            for feat in matched:
                for src in source_map.get(feat, []):
                    if src not in sources:
                        sources.append(src)

            alerts.append(MimicAlert(
                disease_id=disease_id,
                display_name=disease_data["display_name"],
                category=disease_data["category"],
                confidence=confidence,
                matched_features=matched,
                matched_sources=sources,
                avg_diagnostic_delay_months=disease_data["avg_diagnostic_delay_months"],
                red_flags=_filter_red_flags(disease_data["red_flags"], patient_sex),
                description=disease_data["description"]
            ))

    alerts.sort(key=lambda x: x.confidence, reverse=True)
    return alerts
