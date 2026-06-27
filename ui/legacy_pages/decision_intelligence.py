import streamlit as st
import html
import io
import plotly.express as px
import pandas as pd

from agents.xai_agent import (
    generate_shap_explanation,
    get_feature_importance_ranking,
    get_positive_negative_impact,
    plot_impact_chart,
    plot_shap_summary,
)
from ui.components import ai_assistant_panel, glass_panel, glass_panel_small, end_glass_panel, rank_medal, require_dataset
from utils.ai_insights import generate_model_insights
from utils.session_manager import SessionKeys, get_autonomous_result, get_dataframe
from utils.safe_checks import is_present
from utils.styles import render_hero


def _parse_feature_importance(raw_value):
    if isinstance(raw_value, pd.DataFrame):
        return raw_value
    if isinstance(raw_value, dict):
        return pd.DataFrame(list(raw_value.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
    if isinstance(raw_value, str):
        try:
            return pd.read_fwf(io.StringIO(raw_value))
        except Exception:
            try:
                return pd.read_csv(io.StringIO(raw_value))
            except Exception:
                return None
    return None


def _is_classification(problem_type):
    return isinstance(problem_type, str) and "class" in problem_type.lower()


def _is_regression(problem_type):
    return isinstance(problem_type, str) and "regress" in problem_type.lower()


def render():
    render_hero("AI Decision Intelligence", "Advanced explainable AI & model interpretation")

    if not require_dataset():
        return

    glass_panel("Explainability Center", "Inspect model decisions, SHAP insights, and AI trust diagnostics in one premium view.")

    # Check if autonomous pipeline has been run
    autonomous_result = get_autonomous_result()
    if autonomous_result is not None and isinstance(autonomous_result, dict):
        # Display read-only results from autonomous pipeline
        st.info("**Results from Autonomous Data Scientist Pipeline**")
        
        # support both xai_results and explainability_results keys
        xai_results = autonomous_result.get("xai_results") or autonomous_result.get("explainability_results") or {}
        ai_trust = autonomous_result.get("ai_trust_results", {})
        mlops = autonomous_result.get("mlops_results", {})

        feature_importance_raw = xai_results.get("feature_importance")
        feature_importance = _parse_feature_importance(feature_importance_raw)

        if isinstance(xai_results, dict) and xai_results and feature_importance is not None and not feature_importance.empty:
            st.subheader("Model Explainability Report")
            summary = xai_results.get("summary") or xai_results.get("explainability") or "Explainability analysis completed by autonomous pipeline."
            if is_present(summary):
                st.write(summary)

            if not feature_importance.empty:
                st.subheader("Feature Importance Ranking")
                st.dataframe(feature_importance, width="stretch")

            # Show AI trust and MLOps summaries if present
            if ai_trust:
                st.subheader("AI Trust & Fairness")
                st.json(ai_trust)
            if mlops:
                st.subheader("MLOps Readiness")
                st.json(mlops)
        else:
            if isinstance(feature_importance_raw, str):
                st.subheader("Feature Importance")
                st.text_area("Autonomous feature importance output", feature_importance_raw, height=260)
            else:
                st.info("Autonomous pipeline did not generate explainability results in a structured format. Model training required.")
        return

    # Interactive explainability workflow (existing logic)
    if not st.session_state.get(SessionKeys.MODEL_TRAINED):
        st.warning("Train models in **AutoML Studio** first.")
        return

    df = get_dataframe()
    model = st.session_state.get(SessionKeys.BEST_MODEL)
    X = st.session_state.get(SessionKeys.X_DATA)
    best_name = st.session_state.get(SessionKeys.BEST_MODEL_NAME)
    pt = st.session_state.get(SessionKeys.PROBLEM_TYPE)
    results = st.session_state.get(SessionKeys.RESULTS) or {}
    best_score = results.get(best_name, 0)

    st.subheader("Model Card")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", best_name)
    c2.metric("Problem Type", pt)
    fmt = f"{best_score:.2%}" if _is_classification(pt) else f"{best_score:.4f}"
    c3.metric("Score", fmt)

    # Determine a safe, well-formatted confidence percentage to display
    raw_conf = st.session_state.get(SessionKeys.CONFIDENCE_SCORE)
    try:
        if raw_conf is None:
            # Fallback: use best model score as a proxy for confidence
            display_conf = round(float(best_score) * 100.0, 2)
        else:
            display_conf = round(float(raw_conf), 2)
    except Exception:
        display_conf = round(float(best_score) * 100.0, 2)

    c4.metric("Confidence", f"{display_conf:.2f}%")

    if st.button("Generate SHAP Analysis", type="primary"):
        with st.spinner("Computing SHAP values..."):
            try:
                sample = X.sample(min(200, len(X)), random_state=42)
                _, shap_vals = generate_shap_explanation(model, sample)
                importance = get_feature_importance_ranking(shap_vals, list(X.columns))
                impact = get_positive_negative_impact(shap_vals, list(X.columns))

                st.session_state[SessionKeys.SHAP_VALUES] = shap_vals
                st.session_state[SessionKeys.SHAP_IMPORTANCE] = importance
                st.session_state[SessionKeys.SHAP_POSITIVE_NEGATIVE] = impact
                st.session_state[SessionKeys.SHAP_COMPUTED] = True
            except Exception as e:
                st.error(f"SHAP failed: {e}")
                return

    if not st.session_state.get(SessionKeys.SHAP_COMPUTED):
        st.info("Click **Generate SHAP Analysis** to compute explanations.")
        return

    importance = st.session_state.get(SessionKeys.SHAP_IMPORTANCE)
    shap_values = st.session_state.get(SessionKeys.SHAP_VALUES)
    if importance is None or shap_values is None:
        st.error("SHAP analysis results are missing. Re-run SHAP analysis to continue.")
        return
    insights, recs, confidence = generate_model_insights(
        df, pt, best_name, results, importance, st.session_state.get(SessionKeys.TARGET_COLUMN)
    )
    st.session_state[SessionKeys.AI_INSIGHTS] = insights
    st.session_state[SessionKeys.RECOMMENDATIONS] = recs
    st.session_state[SessionKeys.CONFIDENCE_SCORE] = confidence

    st.subheader("Feature Impact Ranking")
    for _, row in importance.head(3).iterrows():
        medal = rank_medal(row["Rank"])
        st.markdown(f"**{medal} {row['Feature']}** — |SHAP| = {row['Mean |SHAP|']:.4f}")

    tab_rank, tab_shap, tab_impact, tab_ai = st.tabs(
        ["Ranking", "SHAP Plot", "Pos/Neg Impact", "AI Insights"]
    )

    with tab_rank:
        st.dataframe(importance, width="stretch", hide_index=True)
        fig = px.bar(
            importance.head(15), x="Mean |SHAP|", y="Feature",
            orientation="h", color="Mean |SHAP|", color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=500)
        st.plotly_chart(
            fig,
            width="stretch",
            key="shap_feature_ranking_chart"
        )
    
    with tab_shap:
        st.plotly_chart(
            plot_shap_summary(
                st.session_state[SessionKeys.SHAP_VALUES],
                list(X.columns)
            ),
            width="stretch",
            key="shap_summary_plot",
        )


    with tab_impact:
        impact = st.session_state.get(SessionKeys.SHAP_POSITIVE_NEGATIVE)
        if impact is not None:
            st.plotly_chart(
                plot_impact_chart(impact),
                width="stretch",
                key="positive_negative_impact_chart"
            )
   
    with tab_ai:
        st.progress(confidence / 100)
        st.metric("AI Confidence", f"{confidence}%")

        # Build a single, beginner-friendly conversational explanation
        fmt_score = f"{best_score:.2%}" if _is_classification(pt) else f"{best_score:.4f}"

        # Top features from SHAP importance
        top_feats = []
        try:
            for _, row in importance.head(5).iterrows():
                feat = html.escape(str(row["Feature"]))
                impact = row.get("Mean |SHAP|", None)
                if impact is not None:
                    top_feats.append(f"{feat} (impact {impact:.4f})")
                else:
                    top_feats.append(feat)
        except Exception:
            top_feats = []

        top_feats_text = ", ".join(top_feats) if top_feats else "Feature impacts are still being summarized."
        conversational = (
            f"Your best model is **{html.escape(str(best_name))}** with a score of **{fmt_score}**. "
            f"The AI confidence score is **{confidence}%**. "
            f"The strongest drivers include: {top_feats_text}."
        )
        glass_panel_small(conversational)

        if insights:
            st.markdown("**Natural Language Insights**")
            for idx, insight in enumerate(insights, 1):
                st.markdown(f"**Insight {idx}:** {insight}")

        if recs:
            st.markdown("**Recommendations**")
            for rec in recs:
                st.markdown(f"- {rec}")
