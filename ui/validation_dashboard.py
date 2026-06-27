"""Validation dashboard rendering for Report Center."""

from __future__ import annotations

from typing import Any, Dict, Optional

import streamlit as st

from utils.session_manager import SessionKeys, get_autonomous_result, has_autonomous_result


def _status_badge(status: str) -> str:
    colors = {
        "PASS": ("#047857", "#ECFDF5"),
        "FAIL": ("#B91C1C", "#FEF2F2"),
        "SKIP": ("#92400E", "#FFFBEB"),
    }
    fg, bg = colors.get(status, ("#475569", "#F8FAFC"))
    return (
        f"<span style='display:inline-flex;padding:4px 10px;border-radius:999px;"
        f"background:{bg};color:{fg};font-weight:800;font-size:0.78rem;'>{status}</span>"
    )


def render_validation_dashboard(report: Optional[Dict[str, Any]] = None) -> None:
    """Render AI Validation Score panel in the Report Center."""
    if report is None:
        report = st.session_state.get(SessionKeys.VALIDATION_REPORT)
    if not report and has_autonomous_result():
        report = get_autonomous_result().get("validation_report") if isinstance(get_autonomous_result(), dict) else None

    if not report:
        st.info("Run the autonomous pipeline to generate the AI validation report.")
        return

    score = int(report.get("overall_score", 0))
    overall = report.get("overall_status", "SKIP")
    st.markdown(
        f"""
        <div class="report-kpi-card" style="--kpi-accent:#6366F1;margin-bottom:1rem;">
            <div class="report-kpi-label">AI Validation Score</div>
            <div class="report-kpi-value">{score}/100 {_status_badge(overall)}</div>
            <div class="report-muted">Generated {report.get('generated_at', '—')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    labels = {
        "dataset_analysis": "Dataset Analysis",
        "problem_type": "Problem Type",
        "model_metrics": "Model Metrics",
        "best_model_selection": "Best Model Selection",
        "explainability": "Explainability",
        "report_accuracy": "Report Accuracy",
    }

    sections = report.get("sections") or {}
    for key, label in labels.items():
        section = sections.get(key) or {}
        status = section.get("status", "SKIP")
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center;padding:0.55rem 0;border-bottom:1px solid rgba(148,163,184,0.15);'>"
            f"<span><strong>{label}</strong></span>{_status_badge(status)}</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Validation details", expanded=False):
        for key, section in sections.items():
            st.markdown(f"**{labels.get(key, key)}** — {section.get('status', 'SKIP')}")
            for check in section.get("checks", []):
                st.markdown(
                    f"- {check.get('label')}: **{check.get('status')}** "
                    f"(expected={check.get('expected')}, actual={check.get('actual')})"
                )
                if check.get("details"):
                    st.caption(str(check["details"]))

    json_path = report.get("json_path")
    md_path = report.get("markdown_path")
    col1, col2 = st.columns(2)
    with col1:
        if json_path:
            st.caption(f"JSON: `{json_path}`")
    with col2:
        if md_path:
            st.caption(f"Markdown: `{md_path}`")
