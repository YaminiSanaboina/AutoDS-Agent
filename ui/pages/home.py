import pandas as pd
import streamlit as st

from utils.session_manager import (
    SessionKeys,
    get_dataframe,
    get_dataset_name,
    has_dataset,
    reset_on_new_dataset,
)
from ui.components import feature_card, metric_card
from ui.components import primary_metric_label
from utils.safe_checks import format_accuracy_display
from utils.health_score import compute_health_score
from utils.styling import render_page_header


def render():
    render_page_header(
        "Dashboard",
        "Your AI-powered data science command center",
    )

    df = get_dataframe()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Dataset Status", "Loaded" if has_dataset() else "None")
    with col2:
        metric_card(
            "Rows",
            f"{df.shape[0]:,}" if has_dataset() else "—",
        )
    with col3:
        metric_card(
            "Columns",
            str(df.shape[1]) if has_dataset() else "—",
        )
    with col4:
        health = compute_health_score(df) if has_dataset() else {"score": "—", "grade": "—"}
        metric_card("Health Score", f"{health['score']}" if has_dataset() else "—")

    st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)

    st.subheader("Platform Modules")
    modules = [
        ("Upload Data", "Import CSV or Excel datasets", ""),
        ("Dataset Workspace", "Profile data with health scoring", ""),
        ("AI Cleaning Center", "Detect issues and auto-clean", ""),
        ("EDA Laboratory", "Interactive exploratory analysis", ""),
        ("AutoML Studio", "Train and compare ML models", ""),
        ("Decision Intelligence", "SHAP explanations & AI insights", ""),
        ("AI Report Center", "Executive PDF reports", ""),
    ]

    row1 = st.columns(4)
    row2 = st.columns(3)
    for idx, (title, desc, icon) in enumerate(modules):
        col = row1[idx] if idx < 4 else row2[idx - 4]
        with col:
            feature_card(title, desc, icon)

    if has_dataset():
        st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)
        st.subheader("Quick Stats")

        health = compute_health_score(df)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Health Grade", health["grade"])
        with c2:
            st.metric("Model Trained", "Yes" if st.session_state.get(SessionKeys.MODEL_TRAINED) else "No")
        with c3:
            st.metric(
                "Best Model",
                st.session_state.get(SessionKeys.BEST_MODEL_NAME) or "—",
            )

        results = st.session_state.get(SessionKeys.RESULTS) or {}
        if st.session_state.get(SessionKeys.MODEL_TRAINED) and results:
            st.markdown("<hr class='section-divider'/>", unsafe_allow_html=True)
            st.subheader("Latest Model Performance")
            best = st.session_state.get(SessionKeys.BEST_MODEL_NAME)
            score = results.get(best, 0)
            problem_type = st.session_state.get(SessionKeys.PROBLEM_TYPE) or "Classification"
            label = primary_metric_label(problem_type)
            formatted = format_accuracy_display(score, problem_type)
            st.info(f"**{best}** — {label}: **{formatted}**")
    else:
        st.info(
            "Upload a dataset to unlock the full platform — profiling, cleaning, "
            "EDA, AutoML, explainability, and automated reporting."
        )
