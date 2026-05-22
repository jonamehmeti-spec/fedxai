"""
deep_diagnosis.py — Stage 2-5 Deep Diagnosis Page

Renders the full diagnostic pipeline after the user clicks "Go Deeper":
  Stage 2: Hidden Mimic Alerts
  Stage 3: Differential Diagnosis Ranking
  Stage 4: Next-Best-Test Recommendations (VoI)
  Stage 5: LLM Clinical Summary + Shadow Mode

Import and call render_deep_diagnosis(stage1_proba, patient_values, model)
from app.py when the session state flag is set.
"""

import io
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from stage2_mimic_detector import detect_mimics, get_common_flags, map_patient_features
from stage3_differential import rank_differentials
from stage4_voi import recommend_tests

CATEGORY_COLORS = {
    "endocrine": "#F59E0B",
    "autoimmune": "#8B5CF6",
    "cardiovascular": "#EF4444",
    "sleep": "#3B82F6",
    "metabolic": "#10B981",
    "hematological": "#EC4899",
}

UNCERTAINTY_COLORS = {
    "low": "#10B981",
    "moderate": "#F59E0B",
    "high": "#EF4444",
}

INVASIVENESS_LABELS = {
    "very_low": "Very Low",
    "low": "Low",
    "moderate": "Moderate",
    "high": "High",
}


def _stage_header(number: int, title: str, subtitle: str = ""):
    """Render a cohesive stage header with numbered pill and fading rule."""
    sub_html = (
        f"<div style='font-size:12px; color:#64748B; margin-top:2px; margin-bottom:12px'>{subtitle}</div>"
        if subtitle else ""
    )
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:14px; margin:36px 0 4px 0'>
        <span style='background:#0A1628; color:#ffffff; font-size:10px; font-weight:800;
                     padding:5px 13px; border-radius:20px; letter-spacing:0.10em; flex-shrink:0'>
            STAGE {number}
        </span>
        <span style='font-size:18px; font-weight:700; color:#0A1628; letter-spacing:-0.01em'>
            {title}
        </span>
        <div style='flex:1; height:1px; background:linear-gradient(90deg,#CBD5E1,transparent); margin-left:4px'></div>
    </div>
    {sub_html}
    """, unsafe_allow_html=True)


def generate_deep_llm_summary(
    stage1_label: str,
    stage1_confidence: float,
    mimic_alerts,
    differentials,
    test_recs,
    shadow_diagnosis: str = None
) -> str:
    """Call Groq LLM with full pipeline context for a comprehensive clinical note."""
    api_key = os.getenv("GROQ_API_KEY", "")

    top_mimics = mimic_alerts[:3]
    top_diff = differentials[:3]
    top_test = test_recs[0] if test_recs else None

    mimic_str = "\n".join([
        f"  - {m.display_name} (confidence: {m.confidence:.0%}, avg delay: {m.avg_diagnostic_delay_months}mo)"
        for m in top_mimics
    ])

    diff_str = "\n".join([
        f"  - {d.display_name}: {d.posterior_probability:.1%} posterior probability"
        for d in top_diff
    ])

    test_str = f"{top_test.display_name} (VoI={top_test.voi_score:.3f})" if top_test else "None"

    shadow_str = f"\nClinician's working diagnosis: {shadow_diagnosis}" if shadow_diagnosis else ""

    if not api_key or api_key == "your_groq_api_key_here":
        return _template_deep_summary(
            stage1_label, stage1_confidence, top_mimics, top_diff, top_test, shadow_diagnosis
        )

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        prompt = f"""You are an advanced clinical decision support AI helping physicians avoid diagnostic errors.

Stage 1 screening result:
- Primary prediction: {stage1_label} ({stage1_confidence:.0%} confidence)

Stage 2 - Hidden mimic conditions detected:
{mimic_str}

Stage 3 - Differential diagnosis ranking:
{diff_str}

Stage 4 - Highest-yield next test: {test_str}
{shadow_str}

Write a structured clinical note (4-5 sentences) that:
1. Acknowledges the common diagnosis but raises the possibility of underdiagnosed mimics
2. Highlights the most clinically significant rare diagnosis to investigate
3. Recommends the single most informative next test and why
4. Notes the typical diagnostic delay for the flagged condition
5. Reminds the clinician this is decision support, not a definitive diagnosis
{"6. Briefly note any discrepancy between the AI finding and the clinician's working diagnosis." if shadow_diagnosis else ""}

Use precise clinical language. Be direct. No bullet points."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    except Exception:
        return _template_deep_summary(
            stage1_label, stage1_confidence, top_mimics, top_diff, top_test, shadow_diagnosis
        )


def _template_deep_summary(
    stage1_label, stage1_confidence, top_mimics, top_diff, top_test, shadow_diagnosis
) -> str:
    top_mimic = top_mimics[0] if top_mimics else None
    top_diag = top_diff[0] if top_diff else None

    base = (
        f"While the primary screening indicates {stage1_label} "
        f"({stage1_confidence:.0%} confidence), the diagnostic pipeline has identified "
        f"potential underdiagnosed mimics warranting further investigation. "
    )

    if top_mimic:
        base += (
            f"{top_mimic.display_name} shows the strongest overlap with the current "
            f"clinical presentation, a condition with an average diagnostic delay of "
            f"{top_mimic.avg_diagnostic_delay_months} months in standard clinical workflows. "
        )

    if top_test:
        base += (
            f"The highest-yield next investigation is {top_test.display_name} "
            f"(sensitivity {top_test.sensitivity:.0%}, specificity {top_test.specificity:.0%}), "
            f"which would most efficiently resolve the current diagnostic uncertainty. "
        )

    if shadow_diagnosis and top_diag:
        if shadow_diagnosis.lower() not in top_diag.display_name.lower():
            base += (
                f"Note: the clinician's working diagnosis of {shadow_diagnosis} "
                f"differs from the system's top differential of {top_diag.display_name}; "
                f"consider whether further workup is indicated. "
            )

    base += "This output is generated by a federated AI system and does not constitute a diagnosis."
    return base


def build_deep_pdf(
    stage1_label: str,
    shadow_diagnosis: str,
    differentials: list,
    test_recs: list,
    clinical_note: str,   # kept in signature for backwards compat, not used in PDF
    shadow_alert: str,    # kept in signature for backwards compat, not used in PDF
) -> bytes:
    """Patient-facing clinical PDF — conditions and tests only, no statistics."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=16*mm, bottomMargin=16*mm,
    )

    NAVY = colors.HexColor("#0A1628")
    BLUE = colors.HexColor("#3B6FD4")
    GREY = colors.HexColor("#64748B")
    LGREY = colors.HexColor("#F1F5F9")

    styles = getSampleStyleSheet()
    def sty(name, **kw):
        return ParagraphStyle(name, parent=styles["Normal"], **kw)

    s_title  = sty("tt", fontSize=18, textColor=colors.white, fontName="Helvetica-Bold", leading=22)
    s_date   = sty("dt", fontSize=9,  textColor=colors.HexColor("#C8D6E8"), fontName="Helvetica", leading=14)
    s_sect   = sty("sc", fontSize=8,  textColor=BLUE, fontName="Helvetica-Bold",
                   spaceBefore=12, spaceAfter=5, letterSpacing=1.4)
    s_body   = sty("bd", fontSize=10, textColor=NAVY, fontName="Helvetica", leading=15, spaceAfter=4)
    s_bold   = sty("bl", fontSize=10, textColor=NAVY, fontName="Helvetica-Bold", leading=15, spaceAfter=4)
    s_item   = sty("it", fontSize=10, textColor=NAVY, fontName="Helvetica", leading=15,
                   leftIndent=8, spaceAfter=3)
    s_small  = sty("sm", fontSize=8.5, textColor=GREY, fontName="Helvetica", leading=13, spaceAfter=2)
    s_footer = sty("ft", fontSize=7.5, textColor=GREY, fontName="Helvetica-Oblique",
                   alignment=TA_CENTER, spaceBefore=4)

    def hr():
        return HRFlowable(width="100%", thickness=0.4,
                          color=colors.HexColor("#E2E8F0"), spaceAfter=5, spaceBefore=5)

    def section(text):
        return Paragraph(text.upper(), s_sect)

    story = []

    # ── Header ───────────────────────────────────────────────────────
    hdr = Table([[
        Paragraph("FedXAI Deep Diagnostic Report", s_title),
        Paragraph(datetime.now().strftime("%d %b %Y  %H:%M"), s_date),
    ]], colWidths=["70%", "30%"])
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), NAVY),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
        ("LEFTPADDING",   (0, 0), (0,  -1), 16),
        ("RIGHTPADDING",  (1, 0), (1,  -1), 16),
        ("ALIGN",         (1, 0), (1,  -1), "RIGHT"),
    ]))
    story.append(hdr)
    story.append(Spacer(1, 6*mm))

    # ── Primary screening ─────────────────────────────────────────────
    story.append(section("Primary Screening Result"))
    is_healthy = stage1_label.lower() == "healthy"
    if not is_healthy:
        story.append(Paragraph(f"<b>Prone to:</b> {stage1_label}", s_body))
    if shadow_diagnosis:
        story.append(Paragraph(
            f"<b>Clinician's working diagnosis:</b> {shadow_diagnosis}", s_body))
    story.append(Spacer(1, 3*mm))
    story.append(hr())

    # ── Conditions to investigate ─────────────────────────────────────
    story.append(section("Conditions to Investigate"))
    story.append(Paragraph(
        "The following conditions share clinical features with the primary presentation "
        "and warrant further investigation:", s_body))
    story.append(Spacer(1, 2*mm))
    if differentials:
        for i, d in enumerate(differentials[:5], 1):
            story.append(Paragraph(
                f"{i}.  <b>{d.display_name}</b>  ({d.category.capitalize()})", s_item))
    story.append(Spacer(1, 4*mm))
    story.append(hr())

    # ── Recommended tests ─────────────────────────────────────────────
    story.append(section("Recommended Tests & Investigations"))
    if test_recs:
        top = test_recs[0]
        story.append(Paragraph(f"<b>Priority test:</b> {top.display_name}", s_bold))
        story.append(Paragraph(
            f"Normal range: {top.normal_range}  ·  Results in {top.result_time_days} day(s)",
            s_small))
        story.append(Spacer(1, 3*mm))
        if len(test_recs) > 1:
            story.append(Paragraph("<b>Additional tests to consider:</b>", s_bold))
            story.append(Spacer(1, 1*mm))
            for rec in test_recs[1:]:
                story.append(Paragraph(f"&bull;  {rec.display_name}", s_item))
    story.append(Spacer(1, 6*mm))

    # ── Footer ────────────────────────────────────────────────────────
    story.append(hr())
    story.append(Paragraph(
        "Generated by FedXAI — Federated &amp; Explainable AI for Chronic Disease Prediction.",
        s_footer
    ))

    doc.build(story)
    return buf.getvalue()


def render_deep_diagnosis(
    stage1_proba: Dict[str, float],
    patient_values: Dict,
    stage1_label: str,
    stage1_confidence: float,
    patient_sex: str = "other",
):
    """Main render function — call this from app.py."""

    st.markdown("---")
    st.markdown(f"""
    <div style='margin-bottom:32px; padding-bottom:20px; border-bottom:1px solid #E8EDF4'>
        <div style='display:inline-flex; align-items:center; gap:7px; font-size:10px;
                    font-weight:800; letter-spacing:0.13em; text-transform:uppercase;
                    color:#3B6FD4; background:#EEF3FB; padding:4px 12px;
                    border-radius:20px; margin-bottom:10px'>
            Stages 2 &nbsp;·&nbsp; 3 &nbsp;·&nbsp; 4 &nbsp;·&nbsp; 5
        </div>
        <div style='font-size:30px; font-weight:800; letter-spacing:-0.8px; line-height:1.15;
                    background:linear-gradient(120deg,#0A1628 60%,#3B6FD4);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    background-clip:text'>
            Deep Diagnostic Analysis
        </div>
        <div style='font-size:14px; color:#64748B; margin-top:6px; line-height:1.5'>
            Federated Shadow Mode — analysing rare and underdiagnosed conditions
            that may present similarly to the primary screening result.
        </div>
    </div>
    """, unsafe_allow_html=True)

    common_flags = get_common_flags(stage1_proba)
    active_features = map_patient_features(patient_values)

    # ── Stage 2: Mimic Alerts ──────────────────────────────────────
    _stage_header(2, "Hidden Mimic Detection",
                  "Conditions that share clinical features with your primary screening result.")

    with st.spinner("Scanning for rare mimics..."):
        mimic_alerts = detect_mimics(stage1_proba, patient_values, min_confidence=0.05, patient_sex=patient_sex)

    if not mimic_alerts:
        st.success("No significant rare disease mimics detected for this presentation.")
        return

    # Inject hover CSS once
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
        .mimic-card { margin-bottom: 10px !important; }
        .mimic-red-flags li { font-size: 12px !important; }
    }
    .mimic-card {
        border-radius: 10px;
        padding: 14px 14px 0 14px;
        margin-bottom: 8px;
        cursor: default;
        transition: box-shadow 0.2s ease, transform 0.15s ease;
        position: relative;
    }
    .mimic-card:hover {
        box-shadow: 0 6px 24px rgba(0,0,0,0.13);
        transform: translateY(-2px);
    }
    .mimic-card-body {
        padding-bottom: 14px;
    }
    .mimic-red-flags {
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.35s ease, opacity 0.3s ease, padding 0.3s ease;
        opacity: 0;
        border-top: 1px solid transparent;
        padding: 0;
    }
    .mimic-card:hover .mimic-red-flags {
        max-height: 300px;
        opacity: 1;
        padding: 10px 0 12px 0;
        border-top-color: rgba(0,0,0,0.07);
    }
    .mimic-red-flags ul {
        margin: 0;
        padding-left: 16px;
    }
    .mimic-red-flags li {
        font-size: 11px;
        color: #374151;
        margin-bottom: 4px;
        line-height: 1.5;
    }
    .mimic-hover-hint {
        font-size: 10px;
        color: #94A3B8;
        margin-top: 6px;
        padding-bottom: 10px;
        letter-spacing: 0.02em;
    }
    .mimic-card:hover .mimic-hover-hint {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)

    # Show mimic cards in columns
    cols = st.columns(min(len(mimic_alerts), 3))
    for i, alert in enumerate(mimic_alerts[:3]):
        with cols[i]:
            cat_color = CATEGORY_COLORS.get(alert.category, "#64748B")

            sources_html = ""
            if alert.matched_sources:
                tags = "".join(
                    f"<span style='display:inline-block; background:{cat_color}18; color:{cat_color}; "
                    f"font-size:10px; font-weight:600; padding:2px 7px; border-radius:12px; "
                    f"margin:2px 3px 2px 0'>{s}</span>"
                    for s in alert.matched_sources
                )
                sources_html = f"<div style='margin-top:8px; line-height:1.8'>{tags}</div>"

            flags_html = "".join(f"<li>{flag}</li>" for flag in alert.red_flags)

            st.markdown(f"""
            <div class='mimic-card' style='border:1px solid {cat_color}; border-left:4px solid {cat_color}; background:#FFFFFF;'>
                <div class='mimic-card-body'>
                    <div style='font-size:11px; color:{cat_color}; font-weight:600; text-transform:uppercase; letter-spacing:0.06em'>
                        {alert.category}
                    </div>
                    <div style='font-size:15px; font-weight:700; margin:4px 0; color:#0A1628'>
                        {alert.display_name}
                    </div>
                    <div style='font-size:22px; font-weight:800; color:{cat_color}'>
                        {alert.confidence:.0%}
                    </div>
                    <div style='font-size:11px; color:#64748B'>overlap confidence</div>
                    {sources_html}
                    <div style='font-size:11px; margin-top:8px; color:#EF4444; font-weight:600'>
                        Avg diagnostic delay: {alert.avg_diagnostic_delay_months} months
                    </div>
                    <div class='mimic-hover-hint'>Hover to see red flags</div>
                </div>
                <div class='mimic-red-flags'>
                    <div style='font-size:10px; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px'>
                        Clinical Red Flags
                    </div>
                    <ul>{flags_html}</ul>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Stage 3: Differential Diagnosis ───────────────────────────
    _stage_header(3, "Differential Diagnosis Ranking",
                  "Bayesian posterior probabilities given current clinical evidence.")

    differentials = rank_differentials(
        mimic_alerts, patient_values, common_flags, active_features
    )

    if differentials:
        diff_df = pd.DataFrame([{
            "Diagnosis": d.display_name,
            "Category": d.category.capitalize(),
            "Probability": f"{d.posterior_probability:.1%}",
            "Uncertainty": d.uncertainty.capitalize(),
            "CI": f"{d.confidence_interval[0]:.1%} – {d.confidence_interval[1]:.1%}",
        } for d in differentials])

        st.dataframe(diff_df, hide_index=True, use_container_width=True)

        fig = go.Figure()
        colors = [CATEGORY_COLORS.get(d.category, "#64748B") for d in differentials]
        fig.add_trace(go.Bar(
            x=[d.posterior_probability * 100 for d in differentials],
            y=[d.display_name for d in differentials],
            orientation="h",
            marker_color=colors,
            text=[f"{d.posterior_probability:.1%}" for d in differentials],
            textposition="outside",
        ))
        fig.update_layout(
            title="Differential Diagnosis — Posterior Probabilities",
            xaxis_title="Probability (%)",
            height=320,
            margin=dict(t=50, b=30, l=10, r=60),
            xaxis=dict(range=[0, max(d.posterior_probability * 100 for d in differentials) * 1.3])
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Stage 4: VoI Test Recommendation ──────────────────────────
    _stage_header(4, "Next-Best-Test Recommendations",
                  "Ranked by Value of Information (VoI) — which test reduces diagnostic uncertainty the most per unit of invasiveness.")

    test_recs = recommend_tests(differentials, top_n=4)

    if test_recs:
        top = test_recs[0]
        st.markdown(f"""
        <div style='background:#0D9488; border-radius:10px; padding:16px; margin-bottom:12px'>
            <div style='font-size:10px; color:#CCFBF1; font-weight:800; letter-spacing:0.10em; text-transform:uppercase'>
                Highest Yield Next Test
            </div>
            <div style='font-size:18px; font-weight:800; color:white; margin:4px 0'>
                {top.display_name}
            </div>
            <div style='font-size:13px; color:#CCFBF1'>
                VoI Score: {top.voi_score:.3f} &nbsp;|&nbsp;
                Sensitivity: {top.sensitivity:.0%} &nbsp;|&nbsp;
                Specificity: {top.specificity:.0%} &nbsp;|&nbsp;
                Invasiveness: {INVASIVENESS_LABELS.get(top.invasiveness, top.invasiveness)} &nbsp;|&nbsp;
                Results in {top.result_time_days} day(s)
            </div>
            <div style='font-size:12px; color:#E0F2FE; margin-top:8px'>
                {top.reasoning}
            </div>
            <div style='font-size:11px; color:#CCFBF1; margin-top:6px'>
                Normal range: {top.normal_range}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if len(test_recs) > 1:
            cols = st.columns(len(test_recs) - 1)
            for i, rec in enumerate(test_recs[1:]):
                with cols[i]:
                    st.markdown(f"""
                    <div style='border:1px solid #CBD5E1; border-radius:8px; padding:12px'>
                        <div style='font-size:13px; font-weight:700'>{rec.display_name}</div>
                        <div style='font-size:11px; color:#64748B; margin-top:4px'>
                            VoI: {rec.voi_score:.3f}<br>
                            Sens: {rec.sensitivity:.0%} / Spec: {rec.specificity:.0%}<br>
                            Invasiveness: {INVASIVENESS_LABELS.get(rec.invasiveness, rec.invasiveness)}<br>
                            {rec.result_time_days} day(s)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Stage 5: Shadow Mode + LLM Summary ────────────────────────
    _stage_header(5, "Clinical Summary & Shadow Mode",
                  "Enter your working diagnosis before generating the AI clinical note.")

    shadow_diagnosis = st.text_input(
        "Your working diagnosis",
        placeholder="e.g. Essential Hypertension, Type 2 Diabetes...",
        help="Enter what you would have diagnosed. The system will compare its findings to yours."
    )

    if shadow_diagnosis.strip():
        generate_btn = st.button("Generate Clinical Note", use_container_width=False)

        if generate_btn:
            with st.spinner("Generating comprehensive clinical note..."):
                summary = generate_deep_llm_summary(
                    stage1_label=stage1_label,
                    stage1_confidence=stage1_confidence,
                    mimic_alerts=mimic_alerts,
                    differentials=differentials,
                    test_recs=test_recs,
                    shadow_diagnosis=shadow_diagnosis,
                )
            st.session_state["deep_clinical_note"] = summary
            st.session_state["deep_shadow_diagnosis"] = shadow_diagnosis
            # Build shadow alert text for PDF
            if differentials:
                top_diff_name = differentials[0].display_name.lower()
                if shadow_diagnosis.lower() not in top_diff_name:
                    st.session_state["deep_shadow_alert"] = (
                        f"Shadow Mode Alert: Your working diagnosis ({shadow_diagnosis}) "
                        f"differs from the system's top differential ({differentials[0].display_name}). "
                        f"Average diagnostic delay: "
                        f"{mimic_alerts[0].avg_diagnostic_delay_months if mimic_alerts else '?'} months."
                    )
                else:
                    st.session_state["deep_shadow_alert"] = (
                        f"Shadow Mode: Your working diagnosis aligns with the system's "
                        f"top differential ({differentials[0].display_name})."
                    )
            else:
                st.session_state["deep_shadow_alert"] = ""
            st.session_state["deep_differentials"] = differentials
            st.session_state["deep_test_recs"] = test_recs

        # Display stored note if available
        if st.session_state.get("deep_clinical_note"):
            summary = st.session_state["deep_clinical_note"]
            st.info(summary)

            shadow_alert = st.session_state.get("deep_shadow_alert", "")
            if shadow_alert.startswith("Shadow Mode Alert"):
                st.warning(f"**{shadow_alert}**")
            elif shadow_alert:
                st.success(f"**{shadow_alert}**")

            st.caption("This system is a decision-support tool. All findings require clinical validation.")

            # ── Download clinical report PDF ───────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            try:
                pdf_bytes = build_deep_pdf(
                    stage1_label=stage1_label,
                    shadow_diagnosis=st.session_state.get("deep_shadow_diagnosis", ""),
                    differentials=st.session_state.get("deep_differentials", differentials),
                    test_recs=st.session_state.get("deep_test_recs", test_recs),
                    clinical_note=summary,
                    shadow_alert=shadow_alert,
                )
                st.download_button(
                    label="Download Clinical Report (PDF)",
                    data=pdf_bytes,
                    file_name="fedxai_clinical_report.pdf",
                    mime="application/pdf",
                    use_container_width=False,
                )
            except Exception as _e:
                st.warning(f"PDF unavailable: {_e}")

    # Diagnostic delay context
    if mimic_alerts:
        with st.expander("Why This Matters — Diagnostic Delay Statistics"):
            delay_df = pd.DataFrame([{
                "Condition": a.display_name,
                "Avg Delay (months)": a.avg_diagnostic_delay_months,
                "Category": a.category.capitalize(),
                "Description": a.description
            } for a in mimic_alerts[:5]])
            st.dataframe(delay_df, hide_index=True, use_container_width=True)
            st.caption(
                "These delays represent real-world averages from published literature. "
                "Earlier detection through systematic screening significantly improves outcomes."
            )
