import pandas as pd
import streamlit as st

from agents.eda_agent import (
    correlation_heatmap,
    detect_outliers,
    generate_eda,
    generate_eda_insights,
    missing_values_chart,
    plot_boxplot,
    plot_histogram,
    plot_pair,
    plot_scatter,
    plot_violin,
)
from ui.components import ai_assistant_panel, dataset_banner, glass_panel, glass_panel_small, end_glass_panel, require_dataset
from utils.llm_insights import generate_eda_insight_llm
from utils.session_manager import SessionKeys, get_autonomous_result, get_dataframe
from utils.styles import render_hero


def render():
    render_hero("EDA Explorer", "Interactive exploratory data analysis laboratory")

    if not require_dataset():
        return

    glass_panel("EDA Intelligence", "Explore data distributions, correlations, and AI insights with confidence.")
    df = get_dataframe()
    dataset_banner()

    # Check if autonomous pipeline has been run
    autonomous_result = get_autonomous_result()
    if autonomous_result is not None and isinstance(autonomous_result, dict):
        # Display read-only results from autonomous pipeline
        st.info("**Results from Autonomous Data Scientist Pipeline**")
        
        eda_results = autonomous_result.get("eda_results", {})
        
        if isinstance(eda_results, dict) and len(eda_results) > 0:
            st.subheader("EDA Summary")
            summary_data = eda_results.get("summary", None)
            if isinstance(summary_data, pd.DataFrame):
                st.dataframe(summary_data, width="stretch")
            elif isinstance(summary_data, str):
                st.text_area("Summary", summary_data, height=320)
            elif isinstance(summary_data, dict):
                st.json(summary_data)
            elif summary_data is not None:
                st.write(summary_data)
            else:
                st.info("No EDA summary available.")
            
            st.subheader("Key Insights")
            insights = eda_results.get("insights", [])
            if isinstance(insights, list) and len(insights) > 0:
                for insight in insights:
                    st.markdown(f"- {insight}")
            
            st.subheader("Visualizations from Autonomous Pipeline")
            # Display charts if available
            charts = eda_results.get("charts", {})
            if isinstance(charts, dict) and "distribution" in charts and charts["distribution"] is not None:
                st.plotly_chart(charts["distribution"], width="stretch")
            if isinstance(charts, dict) and "correlation" in charts and charts["correlation"] is not None:
                st.plotly_chart(charts["correlation"], width="stretch")
        else:
            st.info("Autonomous pipeline did not generate detailed EDA results. Interactive analysis available below.")
    else:
        # Interactive EDA workflow (existing logic)
        if st.button("Run EDA Analysis", type="primary"):
            eda = generate_eda(df)
            st.session_state[SessionKeys.EDA_GENERATED] = True
            st.session_state[SessionKeys.EDA_SUMMARY] = eda["summary"]
            st.session_state[SessionKeys.EDA_NUMERICAL_COLUMNS] = eda["numerical_columns"]
            st.session_state[SessionKeys.EDA_CATEGORICAL_COLUMNS] = eda["categorical_columns"]
            st.session_state[SessionKeys.EDA_INSIGHTS] = generate_eda_insights(df, eda["numerical_columns"])
            if isinstance(eda.get("numerical_columns"), list) and len(eda.get("numerical_columns", [])) > 0 and not st.session_state.get(SessionKeys.EDA_SELECTED_FEATURE):
                st.session_state[SessionKeys.EDA_SELECTED_FEATURE] = eda["numerical_columns"][0]
                if len(eda["numerical_columns"]) > 1:
                    st.session_state[SessionKeys.EDA_SELECTED_FEATURE_2] = eda["numerical_columns"][1]

        if not st.session_state.get(SessionKeys.EDA_GENERATED):
            st.info("Click **Run EDA Analysis** to begin exploration.")
            return

        num_cols = st.session_state.get(SessionKeys.EDA_NUMERICAL_COLUMNS) or []

        tab_overview, tab_dist, tab_corr, tab_outliers, tab_rel = st.tabs(
            ["Overview", "Distributions", "Correlations", "Outliers", "Relationships"]
        )

        with tab_overview:
            st.dataframe(st.session_state[SessionKeys.EDA_SUMMARY], width="stretch")
            st.plotly_chart(missing_values_chart(df), width="stretch")
            st.subheader("AI Insights")

            # Consolidate all EDA insights into a single beginner-friendly message
            eda_insights = st.session_state.get(SessionKeys.EDA_INSIGHTS) or []
            num_numerical = len(st.session_state.get(SessionKeys.EDA_NUMERICAL_COLUMNS) or [])
            num_categorical = len(st.session_state.get(SessionKeys.EDA_CATEGORICAL_COLUMNS) or [])
            selected_feat = st.session_state.get(SessionKeys.EDA_SELECTED_FEATURE)

            feature_insight = ""
            try:
                if selected_feat:
                    from agents.eda_agent import compute_feature_stats

                    stats = compute_feature_stats(df, selected_feat)
                    feature_insight = generate_eda_insight_llm(df, selected_feat, stats)
            except Exception:
                feature_insight = ""

            # Attempt to include outlier info for the selected feature
            outlier_summary = ""
            try:
                if selected_feat:
                    info = detect_outliers(df, selected_feat)
                    outlier_summary = (
                        f"Feature '{selected_feat}' has {info['count']} outliers ({info['pct']:.1f}% of records). "
                        f"IQR bounds: [{info['lower']:.2f}, {info['upper']:.2f}]."
                    )
            except Exception:
                outlier_summary = ""

            # Build a friendly, conversational message
            parts = []
            parts.append("Hello — I ran an exploratory analysis and summarized the key findings for you.")
            parts.append(f"I detected {num_numerical} numerical features and {num_categorical} categorical features.")
            if eda_insights:
                # Join several concise insights into one paragraph
                joined = " ".join(eda_insights[:6])
                parts.append(f"Top findings: {joined}")
            if feature_insight:
                parts.append(f"About '{selected_feat}': {feature_insight}")
            if outlier_summary:
                parts.append(outlier_summary)

            parts.append(
                "What the charts show: check the Distributions tab to see how values spread, "
                "the Correlations tab to spot related features, and Relationships to examine pairwise plots."
            )
            parts.append(
                "If you want help interpreting any specific chart or fixing issues I found, tell me which feature "
                "or click Run EDA Analysis and then pick a feature from the Distributions tab."
            )

            # Present AI insights using the standardized assistant panel
            summary = parts[0] if parts else "EDA run completed"
            details = parts[1:] if len(parts) > 1 else []
            recommendation = (
                "Explore the Distributions, Correlations, Outliers, and Relationships tabs for visual analysis."
            )
            ai_assistant_panel(
                title="EDA Assistant",
                summary=summary,
                details=details,
                recommendation=recommendation,
            )

        with tab_dist:
            if not num_cols:
                st.warning("No numerical columns found.")
            else:
                chart_types = ["Histogram", "Boxplot", "Violin"]
                current_chart = st.session_state.get(SessionKeys.EDA_CHART_TYPE, "Histogram")
                chart_type = st.selectbox(
                    "Chart Type",
                    chart_types,
                    index=chart_types.index(current_chart) if current_chart in chart_types else 0,
                    key="eda_chart_type_select",
                )
                st.session_state[SessionKeys.EDA_CHART_TYPE] = chart_type

                current_feat = st.session_state.get(SessionKeys.EDA_SELECTED_FEATURE) or num_cols[0]
                feature = st.selectbox(
                    "Select Feature",
                    num_cols,
                    index=num_cols.index(current_feat) if current_feat in num_cols else 0,
                    key="eda_dist_feature",
                )
                st.session_state[SessionKeys.EDA_SELECTED_FEATURE] = feature

                from agents.eda_agent import compute_feature_stats
                stats = compute_feature_stats(df, feature)

                if chart_type == "Histogram":
                    st.plotly_chart(plot_histogram(df, feature), width="stretch")
                elif chart_type == "Boxplot":
                    st.plotly_chart(
                        plot_boxplot(df, feature),
                        width="stretch",
                        key=f"boxplot_{feature}"
                    )

                else:
                    st.plotly_chart(plot_violin(df, feature), width="stretch")

        with tab_corr:
            st.plotly_chart(correlation_heatmap(df), width="stretch")

        with tab_outliers:
            if not num_cols:
                st.warning("No numerical columns.")
            else:
                feat = st.selectbox(
                    "Feature for Outlier Analysis",
                    num_cols,
                    index=num_cols.index(st.session_state.get(SessionKeys.EDA_SELECTED_FEATURE, num_cols[0]))
                    if st.session_state.get(SessionKeys.EDA_SELECTED_FEATURE) in num_cols else 0,
                    key="eda_outlier_feature",
                )
                info = detect_outliers(df, feat)
                st.metric("Outliers", info["count"], delta=f"{info['pct']:.1f}%")
                st.write(f"IQR bounds: [{info['lower']:.2f}, {info['upper']:.2f}]")
                st.plotly_chart(plot_boxplot(df, feat), width="stretch")
                # Outlier message is consolidated into the Overview AI assistant

        with tab_rel:
            if len(num_cols) >= 2:
                x_feat = st.selectbox("X Axis", num_cols, key="eda_scatter_x_legacy")
                y_feat = st.selectbox("Y Axis", num_cols, key="eda_scatter_y_legacy")
                st.plotly_chart(plot_scatter(df, x_feat, y_feat), width="stretch")
                pair_fig = plot_pair(df, num_cols)
                if pair_fig:
                    st.plotly_chart(pair_fig, width="stretch")
            else:
                st.warning("Need at least 2 numerical columns for relationship plots.")
