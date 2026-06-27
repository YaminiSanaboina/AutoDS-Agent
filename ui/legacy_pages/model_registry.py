import streamlit as st
import pandas as pd
import plotly.express as px

from agents.model_registry_agent import ModelRegistryAgent
from ui.components import require_dataset, rank_medal, status_badge
from utils.session_manager import get_dataframe, get_dataset_name, SessionKeys
from utils.styles import render_hero


def render():
    render_hero("Model Registry", "Track versions, compare performance, and manage rollbacks")

    registry = ModelRegistryAgent()
    versions = registry.get_model_versions()
    best_model_name = st.session_state.get(SessionKeys.BEST_MODEL_NAME)
    problem_type = st.session_state.get(SessionKeys.PROBLEM_TYPE, "Classification")

    if not versions:
        st.info("No model registry entries found yet. Train models in AutoML Studio to register them.")
        return

    support_cols = st.columns(3)
    support_cols[0].metric("Active Model", best_model_name or "None")
    support_cols[1].metric("Problem Type", problem_type)
    support_cols[2].metric("Registered Versions", len(versions))

    st.markdown("---")
    st.subheader("Leaderboard")
    leaderboard = registry.generate_leaderboard(problem_type)
    if not leaderboard:
        st.info("No leaderboard entries available for the current problem type.")
    else:
        leaderboard_df = pd.DataFrame(leaderboard)
        st.dataframe(leaderboard_df[["rank", "model_name", "version", "dataset", "score", "status"]], width="stretch")
        fig = px.bar(
            leaderboard_df,
            x="score",
            y="model_name",
            orientation="h",
            color="score",
            color_continuous_scale="Blues",
            title="Model Ranking by Composite Score",
        )
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Model Versions")
    grouped = {}
    for entry in versions:
        grouped.setdefault(entry["model_name"], []).append(entry)

    for model_name, entries in grouped.items():
        with st.expander(f"{model_name} ({len(entries)} versions)"):
            version_df = pd.DataFrame(entries)
            st.dataframe(version_df[["model_id", "version", "timestamp", "deployment_status", "is_active"]], width="stretch")
            if st.button(f"Rollback to latest version of {model_name}", key=f"rollback_{model_name}"):
                target = registry.rollback_model(entries[-1]["model_id"])
                if target:
                    st.success(f"Rolled back to {target['version']}.")
                else:
                    st.error("Rollback failed.")

    st.markdown("---")
    st.subheader("Model Registry Insights")
    active_models = [entry for entry in versions if entry.get("is_active")]
    if active_models:
        for entry in active_models:
            status_badge(entry.get("deployment_status", "Not Deployed"), "completed" if entry.get("deployment_status") != "Not Deployed" else "waiting")
            st.write(f"**{entry['model_name']}** — version {entry['version']} · dataset {entry['dataset_name']} · score {entry.get('metrics', {}).get('test_score', 'N/A')}")
    else:
        st.info("No active model versions found.")
