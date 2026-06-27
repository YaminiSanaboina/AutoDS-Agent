import streamlit as st

from agents.eda_agent import (
    correlation_heatmap,
    generate_eda,
    missing_values_chart,
    plot_histogram,
)
from ui.components import dataset_loaded_banner, require_dataset
from utils.session_manager import SessionKeys, get_dataframe, get_dataset_name
from utils.styling import render_page_header


def render():
    render_page_header(
        "EDA Laboratory",
        "Interactive exploratory data analysis with persistent state",
    )

    if not require_dataset():
        return

    df = get_dataframe()
    dataset_loaded_banner(df, get_dataset_name())

    if st.button("Generate EDA Analysis", type="primary"):
        try:
            eda = generate_eda(df)
            st.session_state[SessionKeys.EDA_GENERATED] = True
            st.session_state[SessionKeys.EDA_SUMMARY] = eda["summary"]
            numerical = list(eda["numerical_columns"])
            st.session_state[SessionKeys.EDA_NUMERICAL_COLUMNS] = numerical
            if numerical and not st.session_state.get(SessionKeys.EDA_SELECTED_FEATURE):
                st.session_state[SessionKeys.EDA_SELECTED_FEATURE] = numerical[0]
        except Exception as exc:
            st.error(f"EDA generation failed: {exc}")

    if not st.session_state.get(SessionKeys.EDA_GENERATED):
        st.info("Click **Generate EDA Analysis** to explore your dataset.")
        return

    tab_overview, tab_missing, tab_corr, tab_dist = st.tabs(
        ["Overview", "Missing Values", "Correlation", "Feature Distribution"]
    )

    with tab_overview:
        st.subheader("Statistical Summary")
        st.dataframe(
            st.session_state.get(SessionKeys.EDA_SUMMARY),
            width="stretch",
        )

        st.subheader("Dataset Snapshot")
        st.dataframe(df.describe(include="all").T, width="stretch")

    with tab_missing:
        st.subheader("Missing Values Analysis")
        st.plotly_chart(
            missing_values_chart(df),
            width="stretch",
        )

        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing"]
        missing["Percent"] = (missing["Missing"] / len(df) * 100).round(2)
        st.dataframe(missing, width="stretch", hide_index=True)

    with tab_corr:
        st.subheader("Correlation Matrix")
        numerical_cols = st.session_state.get(SessionKeys.EDA_NUMERICAL_COLUMNS) or []
        if not numerical_cols:
            st.warning("No numerical columns available for correlation analysis.")
        else:
            st.plotly_chart(
                correlation_heatmap(df),
                width="stretch",
            )

    with tab_dist:
        st.subheader("Feature Distribution")
        numerical_cols = st.session_state.get(SessionKeys.EDA_NUMERICAL_COLUMNS) or []

        if not numerical_cols:
            st.warning("No numerical columns available for distribution plots.")
            return

        options = list(numerical_cols)
        current = st.session_state.get(SessionKeys.EDA_SELECTED_FEATURE)
        if current not in options:
            current = options[0]

        selected = st.selectbox(
            "Select Feature",
            options,
            index=options.index(current),
            key="eda_feature_selectbox",
        )
        st.session_state[SessionKeys.EDA_SELECTED_FEATURE] = selected

        st.plotly_chart(
            plot_histogram(df, selected),
            width="stretch",
        )

        col_stats = df[selected].describe()
        st.dataframe(col_stats.to_frame(name=selected), width="stretch")
