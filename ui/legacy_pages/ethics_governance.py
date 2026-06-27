import streamlit as st

from agents.ai_ethics_agent import AIEthicsAgent
from ui.components import ai_assistant_panel, dataset_banner, require_dataset, status_badge, glass_metric
from utils.session_manager import get_dataframe
from utils.styles import render_hero


def render():
    render_hero("Ethics & Governance", "AI fairness, privacy, and trust monitoring")

    if not require_dataset():
        return

    df = get_dataframe()
    dataset_banner()

    ethics_agent = AIEthicsAgent()
    bias_report = ethics_agent.analyze_dataset_bias(df)
    privacy_report = ethics_agent.analyze_privacy_risk(df)
    fairness_report = ethics_agent.evaluate_model_fairness([], [], {})
    ethics_report = ethics_agent.generate_ethics_report(
        bias_report=bias_report,
        fairness_report=fairness_report,
        privacy_report=privacy_report,
    )
    trust = ethics_agent.calculate_ai_governance_score(
        bias_report=bias_report,
        fairness_report=fairness_report,
        privacy_report=privacy_report,
    )

    st.markdown("---")
    cols = st.columns(3)
    with cols[0]:
        glass_metric("⚖️", "Fairness Score", f"{fairness_report.get('fairness_score', 0)} / 100")
    with cols[1]:
        glass_metric("🔒", "Privacy Risk", privacy_report.get('privacy_risk', 'Low'))
    with cols[2]:
        glass_metric("🛡️", "AI Trust Score", f"{trust.get('score', 0)} / 100")

    st.markdown("---")
    st.subheader("Governance Snapshot")
    st.write(ethics_report.get("executive_summary", "Ethics review complete."))

    st.markdown("---")
    st.subheader("Bias & Sensitive Features")
    status_badge(f"Bias risk: {bias_report.get('bias_risk', 'Low')}", "running" if bias_report.get('bias_risk') == 'Medium' else 'completed' if bias_report.get('bias_risk') == 'Low' else 'error')
    if bias_report.get("sensitive_features"):
        st.write("Detected sensitive features:")
        for sensitive in bias_report["sensitive_features"]:
            status_badge(sensitive, "warning")

    if bias_report.get("issues"):
        for issue in bias_report["issues"]:
            st.write(f"- {issue}")

    st.markdown("---")
    st.subheader("Privacy Risk Details")
    status_badge(f"Privacy risk: {privacy_report.get('privacy_risk', 'Low')}", "error" if privacy_report.get('privacy_risk') == 'High' else 'completed')
    if privacy_report.get("detected_identifiers"):
        st.write("Detected identifiers:")
        for identifier in privacy_report["detected_identifiers"]:
            st.write(f"- {identifier}")
    if privacy_report.get("recommendations"):
        st.write("Recommendations:")
        for rec in privacy_report["recommendations"]:
            st.write(f"- {rec}")

    st.markdown("---")
    st.subheader("AI Governance Recommendations")
    if ethics_report.get("compliance_recommendations"):
        for category, items in ethics_report["compliance_recommendations"].items():
            st.markdown(f"**{category.capitalize()}**")
            for item in items:
                st.write(f"- {item}")

    summary = (
        "This page evaluates your dataset for fairness, privacy, and responsible AI readiness. "
        "Use the guidance above to reduce bias, protect private information, and improve overall AI trust."
    )
    ai_assistant_panel(
        title="⚖️ Governance Assistant",
        summary=summary,
        details=[
            "Fairness scores are based on detected sensitive features and group balance.",
            "Privacy alerts identify direct identifiers and high-risk fields in your dataset.",
            "Use the recommended mitigations to prepare your model for production and audits.",
        ],
        recommendation="Review the identified sensitive features, anonymize private data, and keep monitoring governance metrics as your model evolves.",
    )
