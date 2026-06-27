import pandas as pd
import plotly.express as px
import streamlit as st

from agents.model_agent import detect_problem_type, preprocess_data, train_selected_models
from ui.components import dataset_loaded_banner, require_dataset
from utils.session_manager import (
    SessionKeys,
    get_dataframe,
    get_dataset_name,
    store_model_results,
)
from utils.styling import render_page_header


def render():
    render_page_header(
        "AutoML Studio",
        "Automated model training, comparison, and selection",
    )

    if not require_dataset():
        return

    df = get_dataframe()
    dataset_loaded_banner(df, get_dataset_name())

    columns = list(df.columns)
    current_target = st.session_state.get(SessionKeys.TARGET_COLUMN)
    default_idx = columns.index(current_target) if current_target in columns else 0

    target_column = st.selectbox(
        "Target Column",
        columns,
        index=default_idx,
        help="Select the variable you want to predict",
    )
    st.session_state[SessionKeys.TARGET_COLUMN] = target_column

    if st.button("Train Models", type="primary"):
        try:
            X, y, _ = preprocess_data(df, target_column)
            problem_type = detect_problem_type(y)
            results, best_name, best_model, extras = train_selected_models(X, y, problem_type)
            store_model_results(results, best_name, best_model, problem_type, X, y, extras=extras)
            st.session_state[SessionKeys.SHAP_COMPUTED] = False
            st.session_state[SessionKeys.SHAP_IMPORTANCE] = None
            st.session_state[SessionKeys.SHAP_VALUES] = None
            st.session_state[SessionKeys.AI_INSIGHTS] = None
            st.session_state[SessionKeys.REPORT_GENERATED] = False
        except Exception as error:
            st.error(f"Training failed: {error}")
            return

    if not st.session_state.get(SessionKeys.MODEL_TRAINED) or not st.session_state.get(SessionKeys.RESULTS):
        st.info("Select a target column and click **Train Models** to begin.")
        return

    _render_results()


def _render_results():
    results = st.session_state.get(SessionKeys.RESULTS) or {}
    problem_type = st.session_state.get(SessionKeys.PROBLEM_TYPE, "Classification")
    best_name = st.session_state.get(SessionKeys.BEST_MODEL_NAME, "Unknown")
    metric_label = "Accuracy" if problem_type == "Classification" else "R² Score"

    st.success(f"Problem type: **{problem_type}**")

    result_df = pd.DataFrame(
        list(results.items()),
        columns=["Model", metric_label],
    ).sort_values(metric_label, ascending=False)
    result_df["Rank"] = range(1, len(result_df) + 1)
    result_df = result_df[["Rank", "Model", metric_label]]

    if problem_type == "Classification":
        result_df[metric_label] = result_df[metric_label].apply(lambda x: f"{x:.2%}")
    else:
        result_df[metric_label] = result_df[metric_label].apply(lambda x: f"{x:.4f}")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Model Leaderboard")
        st.dataframe(result_df, width="stretch", hide_index=True)

    with c2:
        st.subheader("Performance Comparison")
        chart_df = pd.DataFrame(
            list(results.items()),
            columns=["Model", "Score"],
        )
        fig = px.bar(
            chart_df.sort_values("Score", ascending=True),
            x="Score",
            y="Model",
            orientation="h",
            title=f"{metric_label} by Model",
            color="Score",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, width="stretch")

    best_score = results[best_name]
    formatted = f"{best_score:.2%}" if problem_type == "Classification" else f"{best_score:.4f}"

    st.markdown("---")
    st.subheader("AI Recommendation")
    st.info(
        f"**Recommended Model:** {best_name}  \n"
        f"**{metric_label}:** {formatted}  \n"
        f"This model achieved the highest {metric_label.lower()} among all candidates. "
        f"Proceed to **Decision Intelligence** for SHAP explanations and AI insights."
    )
