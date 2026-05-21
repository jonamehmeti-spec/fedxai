import sys
import json
import io
import os
import warnings
warnings.filterwarnings("ignore")

# Load .env locally; on Streamlit Cloud secrets are injected via st.secrets
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# Bridge Streamlit Cloud secrets → os.environ so all os.getenv() calls work
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

CLASS_NAMES = {
    0: "Healthy",
    1: "Diabetes / Prediabetes",
    2: "High Cardiovascular Risk",
    3: "Hypertension Risk"
}

CLASS_COLORS = {
    0: "#1D9E75",
    1: "#E8593C",
    2: "#BA7517",
    3: "#534AB7",
}

# Gender-specific flags: key → (readable label, clinical note, risk level)
GENDER_FLAGS = {
    "GestationalDiabetes": (
        "Gestational Diabetes",
        "Associated with 7× higher lifetime risk of Type 2 Diabetes and metabolic syndrome.",
        "high"
    ),
    "PCOS": (
        "PCOS",
        "Drives insulin resistance and metabolic dysfunction; elevates cardiovascular and diabetes risk.",
        "high"
    ),
    "IrregularPeriods": (
        "Irregular Menstrual Cycles",
        "May indicate hormonal imbalance or early PCOS; linked to metabolic risk.",
        "moderate"
    ),
    "Preeclampsia": (
        "Preeclampsia History",
        "Doubles lifetime cardiovascular risk; associated with chronic hypertension.",
        "high"
    ),
    "HormonalTherapy": (
        "Hormonal Contraception / HRT",
        "Can elevate blood pressure, alter lipid profile, and increase clotting risk.",
        "moderate"
    ),
    "ErectileDysfunction": (
        "Erectile Dysfunction",
        "Recognised early marker of atherosclerosis; precedes cardiac events by several years.",
        "high"
    ),
    "ProstatIssues": (
        "Prostate / Urinary Tract Issues",
        "Associated with metabolic syndrome and obesity-related hormonal changes.",
        "moderate"
    ),
    "TestosteroneTherapy": (
        "Testosterone Therapy",
        "Suppresses HDL cholesterol, raises hematocrit, and increases cardiovascular risk.",
        "high"
    ),
}

FLAG_COLORS = {"high": "#E8593C", "moderate": "#BA7517"}

st.set_page_config(
    page_title="FedXAI",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
/* ── Base ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0A1628;
    border-right: 1px solid #1E2D45;
}
[data-testid="stSidebar"] * { color: #C8D6E8 !important; }
[data-testid="stSidebar"] .stRadio label {
    font-size: 14px;
    font-weight: 400;
    padding: 6px 0;
    cursor: pointer;
    transition: color 0.15s;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #ffffff !important; }

.sidebar-logo {
    font-size: 22px;
    font-weight: 600;
    color: #ffffff !important;
    letter-spacing: -0.5px;
    margin-bottom: 2px;
}
.sidebar-sub {
    font-size: 12px;
    color: #90AACB !important;
    line-height: 1.5;
    margin-bottom: 24px;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid #1E2D45;
    margin: 16px 0;
}
.sidebar-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: #90AACB !important;
    padding: 4px 0;
}
.sidebar-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #1D9E75;
    flex-shrink: 0;
}

/* ── Page title ───────────────────────────────────────────────── */
.page-header {
    margin-bottom: 32px;
    padding-bottom: 20px;
    border-bottom: 1px solid #E8EDF4;
}
.page-title-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: #3B6FD4;
    background: #EEF3FB;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 10px;
}
.page-title {
    font-size: 30px;
    font-weight: 800;
    color: #0A1628;
    letter-spacing: -0.8px;
    line-height: 1.15;
    margin: 0;
    background: linear-gradient(120deg, #0A1628 60%, #3B6FD4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.page-subtitle {
    font-size: 14px;
    color: #64748B;
    margin-top: 6px;
    line-height: 1.5;
}

/* ── Metric cards ─────────────────────────────────────────────── */
.metric-card {
    background: #ffffff;
    border: 1px solid #E8EDF4;
    border-radius: 10px;
    padding: 20px 24px;
    transition: box-shadow 0.2s;
}
.metric-card:hover { box-shadow: 0 4px 16px rgba(10,22,40,0.08); }
.metric-label {
    font-size: 12px;
    font-weight: 500;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}
.metric-value {
    font-size: 28px;
    font-weight: 600;
    color: #0A1628;
    letter-spacing: -0.5px;
}
.metric-delta {
    font-size: 12px;
    color: #0F7A5A;
    margin-top: 4px;
    font-weight: 500;
}

/* ── Prediction result box ────────────────────────────────────── */
.prediction-card {
    border-radius: 10px;
    padding: 28px 24px;
    text-align: center;
    border: 1px solid transparent;
    transition: box-shadow 0.2s;
    margin-bottom: 16px;
}
.prediction-card:hover { box-shadow: 0 4px 20px rgba(10,22,40,0.10); }
.prediction-label {
    font-size: 22px;
    font-weight: 600;
    margin: 0 0 6px 0;
    letter-spacing: -0.3px;
}
.prediction-confidence {
    font-size: 15px;
    color: #4A5568;
    margin: 0;
    font-weight: 400;
}

/* ── Section cards ────────────────────────────────────────────── */
.section-card {
    background: #ffffff;
    border: 1px solid #E8EDF4;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 16px;
    transition: box-shadow 0.2s;
}
.section-card:hover { box-shadow: 0 4px 16px rgba(10,22,40,0.06); }
.section-title {
    font-size: 14px;
    font-weight: 600;
    color: #0A1628;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 16px;
}

/* ── Clinical report box ──────────────────────────────────────── */
.report-box {
    background: #F8FAFC;
    border-left: 3px solid #4F8EF7;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    font-size: 14px;
    line-height: 1.7;
    color: #2D3748;
    margin: 12px 0;
}
.report-disclaimer {
    font-size: 12px;
    color: #64748B;
    margin-top: 8px;
}

/* ── Buttons ──────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    transition: all 0.15s !important;
    border: 1px solid #E8EDF4 !important;
}
.stButton > button[kind="primary"] {
    background: #0A1628 !important;
    color: #ffffff !important;
    border-color: #0A1628 !important;
}
.stButton > button[kind="primary"]:hover {
    background: #1E2D45 !important;
    border-color: #1E2D45 !important;
    box-shadow: 0 4px 12px rgba(10,22,40,0.2) !important;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #0A1628 !important;
    box-shadow: 0 2px 8px rgba(10,22,40,0.08) !important;
}

/* ── Form inputs ──────────────────────────────────────────────── */
.stNumberInput input, .stSelectbox select, .stTextInput input {
    border-radius: 8px !important;
    border: 1px solid #E8EDF4 !important;
    font-size: 14px !important;
    transition: border-color 0.15s !important;
}
.stNumberInput input:focus, .stSelectbox select:focus {
    border-color: #4F8EF7 !important;
    box-shadow: 0 0 0 3px rgba(79,142,247,0.1) !important;
}

/* ── Input group label ────────────────────────────────────────── */
.input-group-label {
    font-size: 11px;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    margin: 20px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #E8EDF4;
}

/* ── Privacy flow steps ───────────────────────────────────────── */
.flow-step {
    display: flex;
    gap: 16px;
    align-items: flex-start;
    padding: 12px 0;
    border-bottom: 1px solid #F0F4F8;
}
.flow-step:last-child { border-bottom: none; }
.flow-num {
    width: 26px; height: 26px;
    border-radius: 50%;
    background: #0A1628;
    color: #ffffff;
    font-size: 12px;
    font-weight: 600;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.flow-text {
    font-size: 14px;
    color: #2D3748;
    line-height: 1.5;
}
.flow-text strong { color: #0A1628; }

/* ── Table ────────────────────────────────────────────────────── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── Info/alert override ──────────────────────────────────────── */
.stAlert { border-radius: 8px !important; }

/* ── Deep diagnosis separator ─────────────────────────────────── */
.deep-divider {
    border: none;
    border-top: 1px solid #E8EDF4;
    margin: 32px 0 24px 0;
}
.deep-header {
    font-size: 16px;
    font-weight: 600;
    color: #0A1628;
    margin-bottom: 6px;
}
.deep-sub {
    font-size: 13px;
    color: #475569;
    margin-bottom: 20px;
}

/* ── Mobile nav selectbox — hidden on desktop ─────────────────── */
/* ── Mobile ───────────────────────────────────────────────────── */
@media (max-width: 768px) {

    /* ── Sidebar: full-height navy drawer from left ── */
    [data-testid="stSidebar"] {
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        height: 100dvh !important;
        z-index: 1000 !important;
        box-shadow: 4px 0 24px rgba(0,0,0,0.35) !important;
        width: 240px !important;
        min-width: 240px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        height: 100% !important;
        padding-top: 20px !important;
    }

    /* Sidebar nav items — larger tap targets */
    [data-testid="stSidebar"] .stRadio label {
        font-size: 15px !important;
        padding: 10px 4px !important;
        display: block !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0 !important;
    }

    /* Sidebar toggle button — navy pill on left edge */
    [data-testid="stSidebarCollapsedControl"] {
        background: #0A1628 !important;
        border-radius: 0 10px 10px 0 !important;
        width: 32px !important;
        min-height: 52px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 3px 0 12px rgba(0,0,0,0.25) !important;
        border: none !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        position: fixed !important;
    }
    [data-testid="stSidebarCollapsedControl"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        width: 16px !important;
        height: 16px !important;
    }
    /* Collapse arrow inside open sidebar */
    [data-testid="stSidebar"] button[data-testid="stBaseButton-header"] {
        background: #1E2D45 !important;
        border-radius: 6px !important;
        color: #ffffff !important;
    }

    /* Main content: full width, no sidebar offset */
    .block-container {
        padding-top: 1.2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
        margin-left: 0 !important;
    }

    /* Stack columns vertically */
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    /* Page titles */
    .page-title { font-size: 22px !important; }
    .page-subtitle { font-size: 13px !important; }
    .page-title-eyebrow { font-size: 9px !important; }
    .page-header { margin-bottom: 20px !important; padding-bottom: 14px !important; }

    /* Cards */
    .mimic-card { margin-bottom: 12px !important; }
    .prediction-card { padding: 20px 16px !important; }
    .prediction-label { font-size: 18px !important; }
    .prediction-confidence { font-size: 13px !important; }
    .metric-card { padding: 14px 16px !important; }
    .metric-value { font-size: 22px !important; }

    /* Buttons */
    .stButton > button {
        width: 100% !important;
        padding: 12px !important;
        font-size: 15px !important;
    }

    /* Inputs */
    .input-group-label { margin: 14px 0 8px 0 !important; }
    .report-box { padding: 12px 14px !important; font-size: 13px !important; }
    .sidebar-logo { font-size: 18px !important; }

    /* Radio buttons stack vertically */
    .stRadio [role="radiogroup"] { flex-direction: column !important; }

    /* Tables and charts */
    .stDataFrame { overflow-x: auto !important; }
    .js-plotly-plot { width: 100% !important; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_data():
    try:
        with open(DATA_DIR / "features.txt") as f:
            feature_names = [l.strip() for l in f.readlines()]
        scaler = joblib.load(DATA_DIR / "scaler.pkl")
        weights = np.load(str(MODELS_DIR / "global_model_latest.npy"), allow_pickle=True)
        hospital_models = sorted(MODELS_DIR.glob("hospital_*_round*.pkl"))
        model = joblib.load(hospital_models[-1])
        model.coef_ = weights[0]
        model.intercept_ = weights[1]
        return model, scaler, feature_names, True
    except Exception:
        return None, None, [], False


def load_training_history():
    path = LOGS_DIR / "fl_training_history.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def predict_patient(model, scaler, feature_names, patient_values):
    X = np.array([[patient_values.get(f, 0) for f in feature_names]])
    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[0]
    pred_class = int(np.argmax(proba))
    return pred_class, proba, X_scaled


def append_patient_record(patient_values: dict, pred_class: int, hospital_id: str, feature_names: list) -> int:
    """Append one patient record to a hospital CSV. Returns new row count."""
    path = DATA_DIR / f"hospital_{hospital_id}.csv"
    row = {col: patient_values.get(col, 0) for col in feature_names}
    row["disease_class"] = pred_class
    new_row = pd.DataFrame([row])
    existing = pd.read_csv(path)
    updated = pd.concat([existing, new_row], ignore_index=True)
    updated.to_csv(path, index=False)
    return len(updated)


# ── Sidebar ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>FedXAI</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='sidebar-sub'>Federated & Explainable AI<br>for Chronic Disease Prediction</div>",
        unsafe_allow_html=True
    )
    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "Training Metrics",
        "Patient Predictor",
        "Privacy",
        "System Info"
    ], label_visibility="collapsed")

    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='sidebar-badge'><div class='sidebar-dot'></div>No raw data shared</div>
    <div class='sidebar-badge'><div class='sidebar-dot'></div>FedAvg aggregation</div>
    <div class='sidebar-badge'><div class='sidebar-dot'></div>SHAP + LLM explanations</div>
    """, unsafe_allow_html=True)

model, scaler, feature_names, model_ready = load_model_and_data()


# ── Training Metrics ───────────────────────────────────────────────
if "Training Metrics" in page:
    st.markdown("""
    <div class='page-header'>
        <div class='page-title-eyebrow'>Federated Learning</div>
        <div class='page-title'>Training Metrics</div>
        <div class='page-subtitle'>Federated learning performance across rounds</div>
    </div>
    """, unsafe_allow_html=True)

    history = load_training_history()

    if not history:
        st.info("No training history found. Run `python run_simulation.py` to start.")
    else:
        rounds = [h.get("round", i + 1) for i, h in enumerate(history)]
        accuracies = [h.get("accuracy", h.get("avg_accuracy", 0)) for h in history]
        f1s = [h.get("f1_weighted", h.get("avg_f1_weighted", 0)) for h in history]

        acc_delta = (accuracies[-1] - accuracies[0]) * 100
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Final Accuracy</div>
                <div class='metric-value'>{accuracies[-1]*100:.2f}%</div>
                <div class='metric-delta'>+{acc_delta:.2f}% from round 1</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Final F1 Score</div>
                <div class='metric-value'>{f1s[-1]:.4f}</div>
                <div class='metric-delta'>Weighted average</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>FL Rounds</div>
                <div class='metric-value'>{len(history)}</div>
                <div class='metric-delta'>3 hospitals per round</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rounds, y=[a * 100 for a in accuracies],
            mode="lines+markers", name="Accuracy (%)",
            line=dict(color="#534AB7", width=2), marker=dict(size=7, color="#534AB7"),
        ))
        fig.add_trace(go.Scatter(
            x=rounds, y=[f * 100 for f in f1s],
            mode="lines+markers", name="F1 Score (%)",
            line=dict(color="#1D9E75", width=2, dash="dot"), marker=dict(size=7, color="#1D9E75"),
        ))
        fig.update_layout(
            title=None,
            xaxis_title="Round",
            yaxis_title="Score (%)",
            height=320,
            margin=dict(t=20, b=40, l=20, r=20),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=dict(family="Inter", size=13, color="#2D3748"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor="#F0F4F8", zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="#F0F4F8", zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div style='font-size:13px; color:#6B82A0; padding: 8px 0;'>
            Patient records remained on each hospital server. Only model weight arrays were transmitted.
        </div>
        """, unsafe_allow_html=True)


# ── Patient Predictor ──────────────────────────────────────────────
elif "Patient Predictor" in page:
    st.markdown("""
    <div class='page-header'>
        <div class='page-title-eyebrow'>Stage 1 &nbsp;·&nbsp; Screening</div>
        <div class='page-title'>Patient Risk Predictor</div>
        <div class='page-subtitle'>Enter clinical indicators to assess chronic disease risk</div>
    </div>
    """, unsafe_allow_html=True)

    if not model_ready:
        st.error("Model not loaded. Run `python run_simulation.py` first.")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("<div class='input-group-label'>Demographics</div>", unsafe_allow_html=True)
            age_years = st.number_input("Age", min_value=18, max_value=100, value=45, step=1)
            age_brfss = min(13, max(1, (age_years - 18) // 5 + 1))

            sex_label = st.selectbox("Sex", ["Male", "Female", "Other / Prefer not to say"])
            sex = 0 if sex_label == "Male" else 1  # binary for model
            patient_sex = "male" if sex_label == "Male" else ("female" if sex_label == "Female" else "other")

            # Gender-specific clinical questions — shown only for the relevant sex
            gestational_diabetes = 0
            pcos = 0
            irregular_periods = 0
            preeclampsia = 0
            menopause_status = "not_applicable"
            hormonal_therapy = 0
            erectile_dysfunction = 0
            prostate_issues = 0
            testosterone_therapy = 0

            if sex_label == "Female":
                st.markdown("<div class='input-group-label'>Female Health Indicators</div>", unsafe_allow_html=True)
                gestational_diabetes = 1 if st.radio(
                    "History of gestational diabetes", ["No", "Yes"], horizontal=True) == "Yes" else 0
                pcos = 1 if st.radio(
                    "Diagnosed with PCOS (Polycystic Ovary Syndrome)", ["No", "Yes"], horizontal=True) == "Yes" else 0
                irregular_periods = 1 if st.radio(
                    "Irregular menstrual cycles", ["No", "Yes"], horizontal=True) == "Yes" else 0
                preeclampsia = 1 if st.radio(
                    "History of preeclampsia or hypertension during pregnancy", ["No", "Yes"], horizontal=True) == "Yes" else 0
                menopause_opts = ["Pre-menopausal", "Peri-menopausal", "Post-menopausal"]
                menopause_status = st.selectbox("Menopausal status", menopause_opts).lower().replace("-", "_").replace(" ", "_")
                hormonal_therapy = 1 if st.radio(
                    "Currently on hormonal contraception or HRT", ["No", "Yes"], horizontal=True) == "Yes" else 0

            elif sex_label == "Male":
                st.markdown("<div class='input-group-label'>Male Health Indicators</div>", unsafe_allow_html=True)
                erectile_dysfunction = 1 if st.radio(
                    "Erectile dysfunction", ["No", "Yes"], horizontal=True) == "Yes" else 0
                prostate_issues = 1 if st.radio(
                    "Prostate issues or lower urinary tract symptoms", ["No", "Yes"], horizontal=True) == "Yes" else 0
                testosterone_therapy = 1 if st.radio(
                    "Currently on testosterone therapy", ["No", "Yes"], horizontal=True) == "Yes" else 0

            edu_options = [
                "Elementary school",
                "Middle school",
                "High school",
                "University",
                "Postgraduate / Higher education",
            ]
            edu_label = st.selectbox("Education level", edu_options, index=2)
            # Map to BRFSS 1–6 scale
            education = [2, 3, 4, 5, 6][edu_options.index(edu_label)]

            income_options = [
                "Less than €1,000 / month",
                "€1,000 – €2,000 / month",
                "€2,000 – €5,000 / month",
                "More than €5,000 / month",
            ]
            income_label = st.selectbox("Monthly income (EUR)", income_options, index=1)
            # Map to BRFSS 1–8 scale (approximate)
            income = [2, 4, 6, 8][income_options.index(income_label)]

            st.markdown("<div class='input-group-label'>Body Measurements</div>", unsafe_allow_html=True)
            col_h, col_w = st.columns(2)
            with col_h:
                height_cm = st.number_input("Height (cm)", min_value=100, max_value=220, value=170, step=1)
            with col_w:
                weight_kg = st.number_input("Weight (kg)", min_value=30, max_value=250, value=75, step=1)
            bmi = round(weight_kg / ((height_cm / 100) ** 2), 1)
            st.caption(f"BMI: **{bmi}**")

            st.markdown("<div class='input-group-label'>Clinical Indicators</div>", unsafe_allow_html=True)
            high_bp = 1 if st.radio("High Blood Pressure", ["No", "Yes"], horizontal=True) == "Yes" else 0
            high_chol = 1 if st.radio("High Cholesterol", ["No", "Yes"], horizontal=True) == "Yes" else 0
            chol_check = 1 if st.radio("Cholesterol check in past 5 years", ["No", "Yes"], horizontal=True) == "Yes" else 0
            smoker = 1 if st.radio("Smoker (100+ cigarettes lifetime)", ["No", "Yes"], horizontal=True) == "Yes" else 0
            heart_disease = 1 if st.radio("Heart disease / attack history", ["No", "Yes"], horizontal=True) == "Yes" else 0
            stroke = 1 if st.radio("History of stroke", ["No", "Yes"], horizontal=True) == "Yes" else 0
            diff_walk = 1 if st.radio("Difficulty walking or climbing stairs", ["No", "Yes"], horizontal=True) == "Yes" else 0

            st.markdown("<div class='input-group-label'>Lifestyle</div>", unsafe_allow_html=True)
            phys_activity = 1 if st.radio("Physically active in past 30 days (outside of work)", ["No", "Yes"], horizontal=True) == "Yes" else 0
            fruits = 1 if st.radio("Eat fruit 1 or more times per day", ["No", "Yes"], horizontal=True) == "Yes" else 0
            veggies = 1 if st.radio("Eat vegetables 1 or more times per day", ["No", "Yes"], horizontal=True) == "Yes" else 0
            hvy_alcohol_note = "14+ drinks/week for men, 7+ drinks/week for women"
            hvy_alcohol = 1 if st.radio(f"Heavy drinker ({hvy_alcohol_note})", ["No", "Yes"], horizontal=True) == "Yes" else 0

            gen_hlth_label = st.select_slider(
                "General health",
                options=["Excellent", "Very Good", "Good", "Fair", "Poor"]
            )
            gen_hlth = ["Excellent", "Very Good", "Good", "Fair", "Poor"].index(gen_hlth_label) + 1

            st.markdown("<div class='input-group-label'>Wellbeing</div>", unsafe_allow_html=True)

            overall_wellbeing_label = st.select_slider(
                "Overall wellbeing (past month)",
                options=["Very Poor", "Poor", "Moderate", "Good", "Excellent"],
                value="Good",
            )
            # Not a direct BRFSS feature — used for context only, stored as string
            overall_wellbeing = overall_wellbeing_label

            ment_hlth_week = st.number_input(
                "Days mental health was not good (past 7 days)",
                min_value=0, max_value=7, value=0, step=1
            )
            # Scale to 0–30 for model compatibility
            ment_hlth = min(30, round(ment_hlth_week * 30 / 7))

            phys_hlth_week = st.number_input(
                "Days physical health was not good (past 7 days)",
                min_value=0, max_value=7, value=0, step=1
            )
            # Scale to 0–30 for model compatibility
            phys_hlth = min(30, round(phys_hlth_week * 30 / 7))

            st.markdown("<div class='input-group-label'>Health Access</div>", unsafe_allow_html=True)
            healthcare = 1 if st.radio(
                "Has health insurance / access to healthcare",
                ["No", "Yes"], horizontal=True
            ) == "Yes" else 0
            any_healthcare = healthcare
            no_doc_cost = 1 if healthcare == 0 else 0

        with col2:
            patient_values = {
                "Age": age_brfss, "BMI": bmi, "HighBP": high_bp, "HighChol": high_chol,
                "Smoker": smoker, "PhysActivity": phys_activity, "GenHlth": gen_hlth,
                "HeartDiseaseorAttack": heart_disease, "CholCheck": chol_check,
                "Stroke": stroke, "Fruits": fruits, "Veggies": veggies,
                "HvyAlcoholConsump": hvy_alcohol, "AnyHealthcare": any_healthcare,
                "NoDocbcCost": no_doc_cost, "MentHlth": ment_hlth,
                "PhysHlth": phys_hlth, "DiffWalk": diff_walk, "Sex": sex,
                "Education": education, "Income": income,
                # Gender-specific (0 for non-applicable sex)
                "GestationalDiabetes": gestational_diabetes,
                "PCOS": pcos,
                "IrregularPeriods": irregular_periods,
                "Preeclampsia": preeclampsia,
                "MenopauseStatus": menopause_status,
                "HormonalTherapy": hormonal_therapy,
                "ErectileDysfunction": erectile_dysfunction,
                "ProstatIssues": prostate_issues,
                "TestosteroneTherapy": testosterone_therapy,
            }

            if st.button("Run Prediction", use_container_width=True, type="primary"):
                pred_class, proba, X_scaled = predict_patient(model, scaler, feature_names, patient_values)
                label = CLASS_NAMES[pred_class]
                confidence = proba[pred_class]
                color = CLASS_COLORS[pred_class]

                st.session_state.update({
                    "stage1_proba": {CLASS_NAMES[i]: float(p) for i, p in enumerate(proba)},
                    "stage1_label": label,
                    "stage1_confidence": float(confidence),
                    "patient_values": patient_values,
                    "patient_sex": patient_sex,
                    "prediction_done": True,
                    "deep_mode": False,
                    "_pred_class": pred_class,
                    "_proba": proba.tolist(),
                    "_X_scaled": X_scaled.tolist(),
                })

            if st.session_state.get("prediction_done"):
                label = st.session_state["stage1_label"]
                confidence = st.session_state["stage1_confidence"]
                pred_class = st.session_state["_pred_class"]
                proba = np.array(st.session_state["_proba"])
                X_scaled = np.array(st.session_state["_X_scaled"])
                color = CLASS_COLORS[pred_class]

                st.markdown(f"""
                <div class='prediction-card' style='background:{color}12; border-color:{color}40;'>
                    <div class='prediction-label' style='color:{color}'>{label}</div>
                    <div class='prediction-confidence'>{confidence*100:.1f}% model confidence</div>
                </div>
                """, unsafe_allow_html=True)

                prob_df = pd.DataFrame({
                    "Condition": list(CLASS_NAMES.values()),
                    "Probability": proba * 100,
                })
                fig_prob = px.bar(
                    prob_df, x="Probability", y="Condition", orientation="h",
                    color="Condition",
                    color_discrete_sequence=list(CLASS_COLORS.values()),
                )
                fig_prob.update_layout(
                    showlegend=False, height=200,
                    margin=dict(t=10, b=10, l=0, r=10),
                    plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
                    xaxis=dict(title="Probability (%)", showgrid=True, gridcolor="#F0F4F8"),
                    yaxis=dict(title=None),
                    font=dict(family="Inter", size=12),
                )
                st.plotly_chart(fig_prob, use_container_width=True)

                st.markdown("<div class='section-title' style='margin-top:12px'>Feature Contributions</div>",
                            unsafe_allow_html=True)

                # Features the user actually answered
                USER_FEATURES = {
                    "Age", "BMI", "Sex", "Education", "Income",
                    "HighBP", "HighChol", "CholCheck", "Smoker",
                    "HeartDiseaseorAttack", "Stroke", "DiffWalk",
                    "PhysActivity", "Fruits", "Veggies", "HvyAlcoholConsump",
                    "GenHlth", "MentHlth", "PhysHlth",
                    "AnyHealthcare", "NoDocbcCost",
                }

                coef = model.coef_[pred_class]
                sv = coef * X_scaled[0]
                # Filter to only user-answered features, sorted by absolute contribution
                user_idx = [
                    i for i, name in enumerate(feature_names)
                    if name in USER_FEATURES
                ]
                user_idx_sorted = sorted(user_idx, key=lambda i: abs(sv[i]), reverse=True)
                feat_names_top = [feature_names[i] for i in user_idx_sorted]
                feat_vals = [float(sv[i]) for i in user_idx_sorted]

                fig, ax = plt.subplots(figsize=(7, 5))
                colors_bar = [color if v > 0 else "#CBD5E0" for v in feat_vals]
                bars = ax.barh(feat_names_top[::-1], feat_vals[::-1], color=colors_bar[::-1], height=0.55)
                ax.axvline(0, color="#CBD5E0", linewidth=0.8)
                ax.set_xlabel("Contribution score (positive = supports prediction, negative = suggests other condition)", color="#4A5568", fontsize=9)
                ax.set_facecolor("#ffffff")
                fig.patch.set_facecolor("#ffffff")
                ax.tick_params(colors="#4A5568", labelsize=9)
                for spine in ax.spines.values():
                    spine.set_color("#E8EDF4")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                for bar, val in zip(bars, feat_vals[::-1]):
                    ax.text(
                        val + (0.005 if val >= 0 else -0.005),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:+.3f}", va="center",
                        ha="left" if val >= 0 else "right",
                        color="#4A5568", fontsize=8
                    )
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#ffffff")
                buf.seek(0)
                st.image(buf, use_column_width=True)
                plt.close(fig)

                # Build a tailored example from the actual top negative contributor
                neg_pairs = [(n, v) for n, v in zip(feat_names_top, feat_vals) if v < 0]
                if neg_pairs:
                    top_neg_name, top_neg_val = min(neg_pairs, key=lambda x: x[1])
                    readable = top_neg_name.replace("_", " ")
                    if label == "Healthy":
                        example_sentence = (
                            f"For example, <b>{readable}</b> has a negative bar here — this means "
                            f"the model sees it as a factor that leans toward one of the disease categories "
                            f"rather than supporting a healthy profile. It does not mean this factor is "
                            f"irrelevant — it may still warrant clinical attention."
                        )
                    else:
                        example_sentence = (
                            f"For example, <b>{readable}</b> has a negative bar here. "
                            f"This factor is still clinically relevant — it just doesn't fit the "
                            f"<i>{label}</i> pattern as strongly as the other factors shown above it."
                        )
                else:
                    example_sentence = (
                        f"All factors shown here actively support the <i>{label}</i> prediction."
                    )

                st.markdown(f"""
                <div style='font-size:12px; color:#64748B; margin-top:4px; line-height:1.6'>
                    <b>How to read this:</b> Positive bars are factors that support the <i>{label}</i> prediction.
                    Negative bars are factors that don't fit this pattern as strongly —
                    they are still clinically meaningful, just not the primary driver here. {example_sentence}
                </div>
                """, unsafe_allow_html=True)

                # ── Gender-specific risk flags ─────────────────────
                active_gender_flags = [
                    (key, GENDER_FLAGS[key])
                    for key in GENDER_FLAGS
                    if patient_values.get(key, 0) not in (0, "not_applicable", "pre_menopausal")
                ]
                # Menopause post needs separate check (string value)
                if patient_values.get("MenopauseStatus") == "post_menopausal":
                    active_gender_flags.append(("MenopauseStatus", (
                        "Post-Menopausal",
                        "Post-menopausal status shifts lipid and blood pressure profiles, increasing CVD risk.",
                        "moderate"
                    )))

                if active_gender_flags:
                    st.markdown("<div class='section-title' style='margin-top:20px'>Additional Risk Factors</div>",
                                unsafe_allow_html=True)
                    st.markdown("""
                    <div style='font-size:12px; color:#64748B; margin-bottom:10px;'>
                        These factors were not part of the training dataset and do not affect the bar chart above,
                        but they are clinically significant and are used to adjust the deep diagnostic pipeline.
                    </div>
                    """, unsafe_allow_html=True)
                    for key, (flag_label, flag_note, risk_level) in active_gender_flags:
                        flag_color = FLAG_COLORS.get(risk_level, "#64748B")
                        st.markdown(f"""
                        <div style='display:flex; align-items:flex-start; gap:12px;
                                    padding:10px 14px; border-left:3px solid {flag_color};
                                    background:{flag_color}10; border-radius:0 8px 8px 0;
                                    margin-bottom:8px;'>
                            <div>
                                <div style='font-size:13px; font-weight:600; color:{flag_color}'>{flag_label}</div>
                                <div style='font-size:12px; color:#475569; margin-top:2px'>{flag_note}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("<div class='section-title' style='margin-top:20px'>Clinical Summary</div>",
                            unsafe_allow_html=True)

                top_3_features = [
                    {
                        "feature": feat_names_top[i],
                        "shap_value": feat_vals[i],
                        "direction": f"supports {label} prediction" if feat_vals[i] > 0
                                     else f"suggests alternative condition"
                    }
                    for i in range(len(feat_names_top))
                ]
                explanation = {
                    "predicted_label": label,
                    "confidence": float(confidence),
                    "top_features": top_3_features,
                }
                from explain import generate_llm_explanation
                with st.spinner("Generating clinical report..."):
                    report = generate_llm_explanation(explanation)
                st.markdown(f"<div class='report-box'>{report}</div>", unsafe_allow_html=True)
                st.markdown(
                    "<div class='report-disclaimer'>Decision-support tool only. Clinical judgment required.</div>",
                    unsafe_allow_html=True
                )

        if st.session_state.get("prediction_done"):
            st.markdown("<hr class='deep-divider'>", unsafe_allow_html=True)
            st.markdown("<div class='deep-header'>Full Diagnostic Pipeline</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='deep-sub'>Screen for rare and underdiagnosed conditions that may "
                "mimic this presentation.</div>",
                unsafe_allow_html=True
            )
            if st.button("Run Full Diagnostic Pipeline", use_container_width=True, type="primary"):
                st.session_state["deep_mode"] = True

        if st.session_state.get("deep_mode"):
            from deep_diagnosis import render_deep_diagnosis
            render_deep_diagnosis(
                stage1_proba=st.session_state["stage1_proba"],
                patient_values=st.session_state["patient_values"],
                stage1_label=st.session_state["stage1_label"],
                stage1_confidence=st.session_state["stage1_confidence"],
                patient_sex=st.session_state.get("patient_sex", "other"),
            )

        # ── Incremental Training — Append to Hospital Dataset ─────
        if st.session_state.get("prediction_done"):
            st.markdown("---")
            st.markdown(f"""
            <div style='margin-bottom:16px; padding-bottom:16px; border-bottom:1px solid #E8EDF4'>
                <div style='display:inline-flex; align-items:center; gap:7px; font-size:10px;
                            font-weight:800; letter-spacing:0.13em; text-transform:uppercase;
                            color:#3B6FD4; background:#EEF3FB; padding:4px 12px;
                            border-radius:20px; margin-bottom:10px'>
                    Incremental Training
                </div>
                <div style='font-size:22px; font-weight:800; letter-spacing:-0.5px;
                            background:linear-gradient(120deg,#0A1628 60%,#3B6FD4);
                            -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                            background-clip:text'>
                    Save Patient Record
                </div>
                <div style='font-size:13px; color:#64748B; margin-top:4px'>
                    Append this patient to a hospital partition. The expanded dataset will be used
                    the next time the federated learning simulation runs.
                </div>
            </div>
            """, unsafe_allow_html=True)

            save_col, hosp_col, status_col = st.columns([1, 1, 2])

            with hosp_col:
                hospital_choice = st.selectbox(
                    "Hospital partition",
                    ["A", "B", "C"],
                    help="Which hospital's local dataset to append this record to.",
                )

            with save_col:
                st.markdown("<br>", unsafe_allow_html=True)
                save_btn = st.button("Save to Dataset", type="primary", use_container_width=True)

            if save_btn:
                try:
                    new_count = append_patient_record(
                        patient_values=st.session_state["patient_values"],
                        pred_class=st.session_state["_pred_class"],
                        hospital_id=hospital_choice,
                        feature_names=feature_names,
                    )
                    st.session_state["_save_msg"] = ("ok", hospital_choice, new_count)
                except Exception as e:
                    st.session_state["_save_msg"] = ("err", str(e), 0)

            if "_save_msg" in st.session_state:
                kind, arg1, arg2 = st.session_state["_save_msg"]
                if kind == "ok":
                    with status_col:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.success(
                            f"Record saved to Hospital {arg1}. "
                            f"Partition now contains {arg2:,} rows. "
                            f"Re-run `python run_simulation.py` to retrain on the expanded dataset."
                        )
                else:
                    with status_col:
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.error(f"Save failed: {arg1}")


# ── Privacy ────────────────────────────────────────────────────────
elif "Privacy" in page:
    st.markdown("""
    <div class='page-header'>
        <div class='page-title-eyebrow'>Data Governance</div>
        <div class='page-title'>Privacy & Architecture</div>
        <div class='page-subtitle'>How federated learning keeps patient data local</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Hospital Data Summary</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Hospital": ["A", "B", "C"],
            "Patients": ["~84,560", "~84,560", "~84,560"],
            "Data shared": ["Never", "Never", "Never"],
            "Weights shared": ["Yes", "Yes", "Yes"],
        }), hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-card' style='margin-top:16px'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Privacy Parameters</div>", unsafe_allow_html=True)
        st.markdown("""
        | Setting | Value |
        |---|---|
        | Algorithm | FedAvg |
        | DP Mechanism | Gaussian (optional) |
        | DP Budget (ε) | 1.0 |
        | Raw data transmitted | 0 bytes |
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>How FedAvg Works</div>", unsafe_allow_html=True)
        st.markdown("""
        <div class='flow-step'>
            <div class='flow-num'>1</div>
            <div class='flow-text'><strong>Local training</strong> — each hospital trains on its own patient records. No data leaves the server.</div>
        </div>
        <div class='flow-step'>
            <div class='flow-num'>2</div>
            <div class='flow-text'><strong>Weight transmission</strong> — only floating-point weight arrays are sent to the aggregation server.</div>
        </div>
        <div class='flow-step'>
            <div class='flow-num'>3</div>
            <div class='flow-text'><strong>FedAvg aggregation</strong> — global weights computed as <code>w = Σ(n_i / N) × w_i</code>.</div>
        </div>
        <div class='flow-step'>
            <div class='flow-num'>4</div>
            <div class='flow-text'><strong>Global model distributed</strong> — hospitals receive updated weights. No hospital sees another's data.</div>
        </div>
        <div class='flow-step'>
            <div class='flow-num'>5</div>
            <div class='flow-text'><strong>Optional DP</strong> — Gaussian noise added to weights before transmission for formal privacy guarantees.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ── System Info ────────────────────────────────────────────────────
elif "System Info" in page:
    st.markdown("""
    <div class='page-header'>
        <div class='page-title-eyebrow'>About</div>
        <div class='page-title'>System Information</div>
        <div class='page-subtitle'>Components, methods, and run commands</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='section-title'>Stack</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Component": [
                "Federated Learning", "ML Model", "Stage 2 Mimics",
                "Stage 3 Differential", "Stage 4 VoI",
                "LLM Reports", "Dashboard", "Dataset"
            ],
            "Tool": [
                "Flower — FedAvg", "scikit-learn LogisticRegression",
                "Rule-based feature overlap", "Bayesian posterior ranking",
                "Shannon entropy (VoI)", "Groq — llama-3.1-8b-instant",
                "Streamlit + Plotly", "CDC BRFSS 2015"
            ],
        }), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("<div class='section-title'>Knowledge Base — 8 Diseases</div>", unsafe_allow_html=True)
        diseases = [
            ("Endocrine", "Cushing's Syndrome", "38 mo avg diagnosis delay"),
            ("Endocrine", "Addison's Disease", "24 mo avg diagnosis delay"),
            ("Endocrine", "Primary Hyperaldosteronism", "18 mo avg diagnosis delay"),
            ("Endocrine", "Hypothyroidism", "12 mo avg diagnosis delay"),
            ("Autoimmune", "Systemic Lupus (SLE)", "72 mo avg diagnosis delay"),
            ("Autoimmune", "Early Rheumatoid Arthritis", "9 mo avg diagnosis delay"),
            ("Autoimmune", "Sjögren's Syndrome", "48 mo avg diagnosis delay"),
            ("Autoimmune", "Antiphospholipid Syndrome", "36 mo avg diagnosis delay"),
        ]
        for category, name, delay in diseases:
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; align-items:center;
                        padding:8px 0; border-bottom:1px solid #F0F4F8; font-size:13px;'>
                <div>
                    <span style='color:#475569; font-size:11px; text-transform:uppercase;
                                 letter-spacing:0.5px; margin-right:8px'>{category}</span>
                    <span style='color:#0A1628; font-weight:500'>{name}</span>
                </div>
                <span style='color:#64748B; font-size:12px'>{delay}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Run Commands</div>", unsafe_allow_html=True)
    st.code(
        "python3 data/prepare_data.py\n"
        "python3 run_simulation.py --rounds 5\n"
        "streamlit run app.py",
        language="bash"
    )
