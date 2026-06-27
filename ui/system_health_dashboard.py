import json
import os
import streamlit as st

try:
    import psutil
except ImportError:
    psutil = None
from agents.ai_ethics_agent import AIEthicsAgent
from ui.components import require_dataset, glass_metric, status_badge
from utils.session_manager import get_dataframe, get_dataset_name, SessionKeys
from utils.styles import render_hero


def _load_json_history(path: str):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def render():
    render_hero("System Health Dashboard", "Infrastructure status, API health, and AI monitoring")

    if not require_dataset():
        return

    df = get_dataframe()
    st.write(f"**Dataset:** {get_dataset_name()} — {df.shape[0]:,} rows · {df.shape[1]} columns")

    cpu = memory_percent = mem_usage = None
    if psutil is not None:
        try:
            cpu = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            process = psutil.Process()
            mem_usage = process.memory_percent()
            memory_percent = memory.percent
        except Exception as exc:
            st.warning(f"System metrics unavailable: {exc}")
    else:
        st.info("Install optional package `psutil` to view CPU and memory metrics.")

    col1, col2, col3 = st.columns(3)
    col1.metric("CPU Usage", f"{cpu:.1f}%" if cpu is not None else "N/A")
    col2.metric("System Memory", f"{memory_percent:.1f}% used" if memory_percent is not None else "N/A")
    col3.metric("App Memory", f"{mem_usage:.1f}%" if mem_usage is not None else "N/A")

    api_health = "Healthy" if st.session_state.get("deployment_ready") else "Pending"
    agent_health = "Good" if st.session_state.get(SessionKeys.MODEL_TRAINED) else "Review"
    model_health = "Stable" if st.session_state.get(SessionKeys.MODEL_TRAINED) else "Needs Training"
    security_alerts = "None" if st.session_state.get(SessionKeys.SHAP_COMPUTED) else "Review pending"
    drift_history = _load_json_history("drift_history.json")
    drift_alerts = "Detected" if drift_history else "None"

    st.markdown("---")
    status_badge(f"API health: {api_health}", "completed" if api_health == "Healthy" else "waiting")
    status_badge(f"Agent health: {agent_health}", "completed" if agent_health == "Good" else "waiting")
    status_badge(f"Model health: {model_health}", "completed" if model_health == "Stable" else "waiting")
    status_badge(f"Security alerts: {security_alerts}", "completed" if security_alerts == "None" else "running")
    status_badge(f"Drift alerts: {drift_alerts}", "error" if drift_alerts == "Detected" else "completed")

    st.markdown("---")
    st.subheader("Monitoring Insights")
    glass_metric("", "CPU", f"{cpu:.1f}%" if cpu is not None else "N/A")
    glass_metric("", "Memory", f"{memory_percent:.1f}%" if memory_percent is not None else "N/A")
    glass_metric("", "Security", security_alerts)
    glass_metric("", "Drift", drift_alerts)

    st.markdown("---")
    st.subheader("System Alerts")
    if drift_history:
        st.warning(f"Drift events found: {len(drift_history)}. Review latest production data drift history.")
    else:
        st.success("No drift alerts detected.")

    if not st.session_state.get(SessionKeys.MODEL_TRAINED):
        st.info("Train a model and run explainability to improve AI health metrics.")
