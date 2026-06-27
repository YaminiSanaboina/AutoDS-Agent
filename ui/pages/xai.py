import pandas as pd
import plotly.express as px
import streamlit as st

from agents.xai_agent import (
    generate_shap_explanation,
    get_feature_importance_ranking,
    plot_shap_summary,
)
from ui.components import insight_box, require_dataset
from utils.ai_insights import generate_model_insights
from utils.session_manager import SessionKeys, get_dataframe
from utils.styling import render_page_header


def render():
    render_page_header(
        "AI Decision Intelligence",
        "SHAP explainability with natural language AI insights",
    )

    if not require_dataset():
        return

    if not st.session_state.get(SessionKeys.MODEL_TRAINED):
        st.warning("Train a model in **AutoML Studio** before viewing explanations.")
        return

    model = st.session_state.get(SessionKeys.BEST_MODEL)
    X = st.session_state.get(SessionKeys.X_DATA, pd.DataFrame())
    best_name = st.session_state.get(SessionKeys.BEST_MODEL_NAME, "N/A")
    problem_type = st.session_state.get(SessionKeys.PROBLEM_TYPE, "Classification")
    results = st.session_state.get(SessionKeys.RESULTS) or {}
    df = get_dataframe() or pd.DataFrame()

    st.subheader("Model Information")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model", best_name)
    c2.metric("Problem Type", problem_type)
    c3.metric("Features", X.shape[1] if not X.empty else 0)
    c4.metric("Samples", f"{X.shape[0]:,}" if not X.empty else "0")

    if st.button("Generate SHAP Analysis", type="primary"):
        with st.spinner("Computing SHAP values..."):
            try:
                sample_size = min(200, len(X))
                X_sample = X.sample(n=sample_size, random_state=42)
                _, shap_values = generate_shap_explanation(model, X_sample)

                importance = get_feature_importance_ranking(
                    shap_values,
                    list(X.columns),
                )
                st.session_state[SessionKeys.SHAP_VALUES] = shap_values
                st.session_state[SessionKeys.SHAP_IMPORTANCE] = importance
                st.session_state[SessionKeys.SHAP_COMPUTED] = True
            except Exception as error:
                st.error(f"SHAP analysis failed: {error}")
                return

    if st.session_state.get(SessionKeys.SHAP_COMPUTED):
        importance = st.session_state.get(SessionKeys.SHAP_IMPORTANCE)

        insights, recommendations, confidence = generate_model_insights(
            df,
            problem_type,
            best_name,
            results,
            importance,
            st.session_state.get(SessionKeys.TARGET_COLUMN),
        )
        st.session_state[SessionKeys.AI_INSIGHTS] = insights
        st.session_state[SessionKeys.RECOMMENDATIONS] = recommendations
        st.session_state[SessionKeys.CONFIDENCE_SCORE] = confidence

        tab_ranking, tab_chart, tab_insights = st.tabs(
            ["Feature Ranking", "SHAP Visualization", "AI Insights"]
        )

        with tab_ranking:
            st.subheader("SHAP Feature Importance Ranking")
            st.dataframe(importance, width="stretch", hide_index=True)

            fig = px.bar(
                importance.head(15),
                x="Mean |SHAP|",
                y="Feature",
                orientation="h",
                title="Top 15 Features by Mean |SHAP|",
                color="Mean |SHAP|",
                color_continuous_scale="Blues",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
            st.plotly_chart(fig, width="stretch")

        with tab_chart:
            st.subheader("SHAP Summary Plot")
            fig = plot_shap_summary(
                st.session_state.get(SessionKeys.SHAP_VALUES),
                list(X.columns),
            )
            st.plotly_chart(fig, width="stretch")

        with tab_insights:
            st.subheader("AI Confidence Score")
            st.progress(confidence / 100)
            st.metric("Confidence", f"{confidence}%")

            st.subheader("Natural Language Insights")
            for idx, insight in enumerate(insights, 1):
                insight_box(f"Insight {idx}", insight)

            st.subheader("Recommendations")
            for rec in recommendations:
                st.write(f"• {rec}")
    else:
        st.info("Click **Generate SHAP Analysis** to compute feature explanations.")
