import datetime
import json
import os
import streamlit as st

from ui.components import require_dataset, status_badge
from utils.session_manager import get_dataframe, get_dataset_name, SessionKeys
from utils.styles import render_hero


def _load_json_history(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _format_time(value):
    if not value:
        return "Unknown"
    try:
        if isinstance(value, str):
            return datetime.datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(value)
    return str(value)


def render():
    render_hero("Agent Activity Monitor", "Track AI agent execution, confidence, and decisions")

    if not require_dataset():
        return

    df = get_dataframe()
    st.write(f"**Dataset:** {get_dataset_name()} — {df.shape[0]:,} rows · {df.shape[1]} columns")

    agents = [
        {"name": "Dataset Intelligence Agent", "status": "COMPLETED", "confidence": "92%", "action": "Analyzed dataset health and problem type."},
        {"name": "Feature Engineering Agent", "status": "RUNNING" if st.session_state.get(SessionKeys.EDA_GENERATED) else "WAITING", "confidence": "80%", "action": "Preparing feature strategy."},
        {"name": "AutoML Agent", "status": "COMPLETED" if st.session_state.get(SessionKeys.MODEL_TRAINED) else "WAITING", "confidence": "85%", "action": "Training candidate models."},
        {"name": "Explainability Agent", "status": "COMPLETED" if st.session_state.get(SessionKeys.SHAP_COMPUTED) else "WAITING", "confidence": "90%", "action": "Computing SHAP explanations."},
        {"name": "Ethics Agent", "status": "COMPLETED", "confidence": "88%", "action": "Evaluated bias and privacy risk."},
        {"name": "Deployment Agent", "status": "WAITING", "confidence": "82%", "action": "Packaging model for production."},
    ]

    st.markdown("---")
    for agent in agents:
        status_badge(agent["status"], "completed" if agent["status"] == "COMPLETED" else "running" if agent["status"] == "RUNNING" else "waiting")
        st.markdown(f"**{agent['name']}** — {agent['confidence']}<br><em>{agent['action']}</em>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Decision Logs")
    history_files = [
        ("storage/monitoring/chief_decisions.json", "Chief Data Scientist"),
        ("storage/memory/auto_ds_memory.json", "Experiment Memory"),
        ("drift_history.json", "Drift Monitoring"),
    ]

    for path, label in history_files:
        records = _load_json_history(path)
        if records:
            with st.expander(f"{label} ({len(records)})"):
                for record in records[-5:][::-1]:
                    st.markdown(f"- {record.get('timestamp', 'Unknown')} — {record.get('decision', record.get('summary', str(record)))}")
        else:
            st.info(f"No records found for {label}.")

    st.markdown("---")
    st.subheader("Live Agent Confidence")
    if st.session_state.get(SessionKeys.CONFIDENCE_SCORE) is not None:
        st.metric("Overall AI Confidence", f"{st.session_state.get(SessionKeys.CONFIDENCE_SCORE)}%")
    else:
        st.info("Confidence scores will appear after model training and explainability analysis.")
