import os

import streamlit as st

from agents.report_agent import generate_pdf_report
from ui.components import require_dataset
from utils.ai_insights import generate_executive_summary, generate_model_insights
from utils.health_score import compute_health_score, detect_data_issues
from utils.styling import render_page_header


def render():
    render_page_header(
        "AI Report Center",
        "Professional executive reports with AI-powered insights",
    )

    if not require_dataset():
        return

    df = st.session_state.df
    health = compute_health_score(df)
    issues = detect_data_issues(df)

    if not st.session_state.model_trained:
        st.warning("Train a model in **AutoML Studio** to include performance metrics in the report.")

    if st.button("Generate AI Report", type="primary"):
        _build_and_generate_report(df, health, issues)

    if st.session_state.report_generated and st.session_state.report_payload:
        payload = st.session_state.report_payload

        st.success("Report generated successfully!")

        tab_exec, tab_quality, tab_model, tab_recs = st.tabs(
            ["Executive Summary", "Dataset Quality", "Model Performance", "Recommendations"]
        )

        with tab_exec:
            st.markdown(payload["executive_summary"])

        with tab_quality:
            st.metric("Health Score", f"{health['score']}/100 — {health['grade']}")
            st.write(health["summary"])
            st.subheader("Quality Breakdown")
            for label, value in health["breakdown"].items():
                st.write(f"**{label}:** {value}/100")
            st.subheader("Detected Issues")
            for issue in issues[:8]:
                st.write(f"- **{issue['title']}:** {issue['description']}")

        with tab_model:
            if st.session_state.model_trained:
                st.write(f"**Problem Type:** {st.session_state.problem_type}")
                st.write(f"**Best Model:** {st.session_state.best_model_name}")
                st.write(f"**Target:** {st.session_state.target_column}")
                st.write(f"**Confidence:** {st.session_state.confidence_score or 'N/A'}%")
                st.subheader("All Models")
                for name, score in st.session_state.results.items():
                    fmt = f"{score:.2%}" if st.session_state.problem_type == "Classification" else f"{score:.4f}"
                    st.write(f"- {name}: {fmt}")
            else:
                st.info("No model results available.")

        with tab_recs:
            recs = payload.get("recommendations", [])
            for rec in recs:
                st.write(f"• {rec}")

        if st.session_state.report_path and os.path.exists(st.session_state.report_path):
            with open(st.session_state.report_path, "rb") as pdf_file:
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_file.read(),
                    file_name="AutoDS_Platform_Report.pdf",
                    mime="application/pdf",
                    type="primary",
                )


def _build_and_generate_report(df, health, issues):
    insights, recommendations, confidence = [], [], 0.0

    if st.session_state.model_trained and st.session_state.results:
        if not st.session_state.ai_insights:
            insights, recommendations, confidence = generate_model_insights(
                df,
                st.session_state.problem_type,
                st.session_state.best_model_name,
                st.session_state.results,
                st.session_state.shap_importance,
                st.session_state.target_column,
            )
        else:
            insights = st.session_state.ai_insights
            recommendations = st.session_state.recommendations
            confidence = st.session_state.confidence_score or 0.0

    executive = generate_executive_summary(
        df,
        health,
        st.session_state.problem_type or "N/A",
        st.session_state.best_model_name or "N/A",
        st.session_state.results or {},
        confidence,
    )

    issue_lines = [
        f"{i['title']}: {i['description']} → {i['recommendation']}"
        for i in issues[:10]
    ]

    model_lines = []
    if st.session_state.results:
        for name, score in st.session_state.results.items():
            fmt = (
                f"{score:.2%}"
                if st.session_state.problem_type == "Classification"
                else f"{score:.4f}"
            )
            model_lines.append(f"{name}: {fmt}")

    shap_lines = []
    if st.session_state.shap_importance is not None:
        for _, row in st.session_state.shap_importance.head(10).iterrows():
            shap_lines.append(
                f"{row['Feature']}: {row['Mean |SHAP|']:.4f}"
            )

    payload = {
        "executive_summary": executive,
        "recommendations": recommendations,
        "dataset_name": st.session_state.upload_filename or "Dataset",
        "rows": df.shape[0],
        "columns": df.shape[1],
        "health_score": health["score"],
        "health_grade": health["grade"],
        "health_summary": health["summary"],
        "health_breakdown": health["breakdown"],
        "issues": issue_lines,
        "problem_type": st.session_state.problem_type or "Not trained",
        "best_model": st.session_state.best_model_name or "Not trained",
        "target_column": st.session_state.target_column or "N/A",
        "confidence": confidence,
        "model_results": model_lines,
        "ai_insights": insights,
        "shap_ranking": shap_lines,
        "cleaning_actions": st.session_state.cleaning_report or ["No cleaning performed"],
    }

    report_path = generate_pdf_report(payload)
    st.session_state.report_payload = payload
    st.session_state.report_path = report_path
    st.session_state.report_generated = True
    st.session_state.ai_insights = insights
    st.session_state.recommendations = recommendations
    st.session_state.confidence_score = confidence
