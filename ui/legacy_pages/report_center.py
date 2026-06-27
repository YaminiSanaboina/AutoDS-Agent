import os
import pandas as pd

import streamlit as st

from agents.report_agent import generate_pdf_report
from ui.components import ai_chat_message, glass_panel, glass_panel_small, end_glass_panel, require_dataset
from utils.ai_insights import (
    generate_executive_summary,
    generate_model_insights,
    generate_beginner_report_sections,
)
from utils.health_score import compute_health_score, detect_data_issues
from utils.llm_insights import generate_dataset_summary_llm
from utils.session_manager import SessionKeys, get_autonomous_result, get_dataframe, get_dataset_name
from utils.styles import render_hero


def render():
    render_hero("AI Research Report", "Professional downloadable analysis reports")

    if not require_dataset():
        return

    glass_panel("Report Overview", "Generate professional AI research reports for your dataset, model, and validation results.")
    df = get_dataframe()
    
    # Check if autonomous pipeline has been run
    autonomous_result = get_autonomous_result()
    if autonomous_result is not None and isinstance(autonomous_result, dict):
        # Display read-only results from autonomous pipeline
        st.info("**Results from Autonomous Data Scientist Pipeline**")

        # Support both dataset_report and dataset_analysis keys
        dataset_report = autonomous_result.get("dataset_report") or autonomous_result.get("dataset_analysis") or {}
        hyperparameter_report = autonomous_result.get("hyperparameter_report", {})
        deployment_readiness = autonomous_result.get("deployment_readiness", {})
        ethics_report = autonomous_result.get("ethics_report", {})
        model_comparison = autonomous_result.get("model_comparison", [])
        improvement_history = autonomous_result.get("improvement_history", [])
        ai_trust = autonomous_result.get("ai_trust_results", {})
        explainability = autonomous_result.get("explainability_results") or autonomous_result.get("xai_results") or {}

        st.subheader("Executive Summary")
        st.write(f"**Project Goal:** {autonomous_result.get('project_goal', 'Not specified')}")
        st.write(f"**Dataset:** {autonomous_result.get('dataset_name', 'Unknown')}")
        st.write(f"**Final Score:** {autonomous_result.get('final_score', '—')} / 100")
        st.write(f"**Recommendation:** {autonomous_result.get('recommendation', 'Review manually')}")

        st.subheader("Dataset Analysis")
        shape = dataset_report.get("dataset_shape", {})
        rows = shape.get("rows") if isinstance(shape, dict) else (shape[0] if isinstance(shape, (list, tuple)) and len(shape) > 0 else "—")
        cols = shape.get("columns") if isinstance(shape, dict) else (shape[1] if isinstance(shape, (list, tuple)) and len(shape) > 1 else "—")
        st.write(f"**Rows:** {rows} | **Columns:** {cols}")

        problem = dataset_report.get("problem_analysis", {})
        st.write(f"**Problem Type:** {problem.get('problem_type', 'Unknown')}")
        st.write(f"**Summary:** {problem.get('summary', problem.get('reason', 'Analysis complete.'))}")

        st.subheader("Model Strategy & Comparison")
        st.write(f"**Recommended Algorithm:** {hyperparameter_report.get('algorithm', 'Unknown')}")
        st.json(hyperparameter_report.get("recommended_parameters", {}))

        if model_comparison:
            st.subheader("Model Comparison")
            rows = []
            for m in model_comparison:
                name = m.get('version') or m.get('model') or m.get('name')
                metrics = m.get('metrics', {})
                rows.append({
                    'Model': name,
                    'Accuracy': metrics.get('accuracy'),
                    'Precision': metrics.get('precision'),
                    'Recall': metrics.get('recall'),
                    'F1': metrics.get('f1'),
                    'ROC-AUC': metrics.get('roc_auc'),
                })
            st.dataframe(pd.DataFrame(rows).fillna('—'), width="stretch")

        if improvement_history:
            st.subheader("AI Improvements Timeline")
            for ev in improvement_history:
                st.write(ev)

        st.subheader("Deployment Readiness")
        st.write(f"**Risk Level:** {deployment_readiness.get('risk_level', 'Unknown')}")
        st.write(f"**Reasoning:** {deployment_readiness.get('reasoning', deployment_readiness.get('warnings', 'N/A'))}")

        st.subheader("Ethics & Fairness")
        st.write(ethics_report.get("executive_summary", "Ethics review completed."))

        st.subheader("AI Trust & Explainability")
        st.json(ai_trust)
        feature_importance = explainability.get('feature_importance')
        if isinstance(feature_importance, (list, dict)):
            st.dataframe(feature_importance, width="stretch")
        elif isinstance(feature_importance, str):
            st.text_area("Feature Importance", feature_importance, height=280)
        else:
            st.write("No structured feature importance available.")

        st.markdown("---")
        st.success("✅ Full autonomous pipeline execution completed. Review outputs above for detailed insights.")
        return

    # Manual report generation (existing logic)
    health = compute_health_score(df)
    issues = detect_data_issues(df)

    if st.button("Generate AI Research Report", type="primary"):
        _build_report(df, health, issues)

    if not st.session_state.get(SessionKeys.REPORT_GENERATED):
        st.info("Click **Generate AI Research Report** to create your PDF.")
        return

    payload = st.session_state.get(SessionKeys.REPORT_PAYLOAD) or {}

    tabs = st.tabs([
        "Executive Summary", "Dataset Analysis", "Data Quality",
        "Model Performance", "Explainable AI", "Recommendations",
    ])

    with tabs[0]:
        st.markdown(payload.get("executive_summary", ""), unsafe_allow_html=True)
        ai_chat_message(payload.get("dataset_summary", ""))

    with tabs[1]:
        st.write(f"**Dataset:** {get_dataset_name()}")
        st.write(f"**Rows:** {df.shape[0]:,} · **Columns:** {df.shape[1]}")
        if payload.get("dataset_summary"):
            st.markdown(payload.get("dataset_summary"), unsafe_allow_html=True)

    with tabs[2]:
        st.metric("Health Score", f"{health['score']}/100")
        for issue in issues[:8]:
            st.write(f"- {issue['title']}: {issue['description']}")

    with tabs[3]:
        if st.session_state.get(SessionKeys.MODEL_TRAINED):
            for name, score in (st.session_state.get(SessionKeys.RESULTS) or {}).items():
                pt = st.session_state.get(SessionKeys.PROBLEM_TYPE)
                fmt = f"{score:.2%}" if pt == "Classification" else f"{score:.4f}"
                st.write(f"**{name}:** {fmt}")
        else:
            st.info("No models trained yet.")

    with tabs[4]:
        imp = st.session_state.get(SessionKeys.SHAP_IMPORTANCE)
        if imp is not None:
            st.dataframe(imp.head(10), width="stretch", hide_index=True)
        else:
            st.info("Run SHAP analysis in Decision Intelligence first.")

    with tabs[5]:
        for rec in payload.get("recommendations", []):
            st.markdown(f"- {rec}", unsafe_allow_html=True)

    path = st.session_state.get(SessionKeys.REPORT_PATH)
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            st.download_button(
                "Download PDF Report",
                f.read(),
                file_name="AutoDS_Agent_Report.pdf",
                mime="application/pdf",
                type="primary",
            )


def _build_report(df, health, issues):
    insights, recs, confidence = [], [], 0.0
    if st.session_state.get(SessionKeys.MODEL_TRAINED):
        insights, recs, confidence = generate_model_insights(
            df,
            st.session_state.get(SessionKeys.PROBLEM_TYPE),
            st.session_state.get(SessionKeys.BEST_MODEL_NAME),
            st.session_state.get(SessionKeys.RESULTS) or {},
            st.session_state.get(SessionKeys.SHAP_IMPORTANCE),
            st.session_state.get(SessionKeys.TARGET_COLUMN),
        )

    shap_lines = []
    imp = st.session_state.get(SessionKeys.SHAP_IMPORTANCE)
    if imp is not None:
        for _, row in imp.head(10).iterrows():
            shap_lines.append(f"{row['Feature']}: {row['Mean |SHAP|']:.4f}")

    model_lines = []
    results = st.session_state.get(SessionKeys.RESULTS) or {}
    for name, score in results.items():
        pt = st.session_state.get(SessionKeys.PROBLEM_TYPE)
        fmt = f"{score:.2%}" if pt == "Classification" else f"{score:.4f}"
        model_lines.append(f"{name}: {fmt}")

    if not model_lines:
        model_lines = ["No trained model results available yet."]

    eda_insights = st.session_state.get(SessionKeys.EDA_INSIGHTS) or []

    model_info = {
        "problem_type": st.session_state.get(SessionKeys.PROBLEM_TYPE),
        "best_model": st.session_state.get(SessionKeys.BEST_MODEL_NAME),
        "model_results": model_lines,
        "target_column": st.session_state.get(SessionKeys.TARGET_COLUMN),
        "confidence": confidence,
        "shap_ranking": shap_lines,
    }

    executive, dataset_summary = generate_beginner_report_sections(
        df, health, issues, model_info=model_info, eda_findings=eda_insights
    )

    payload = {
        "executive_summary": executive,
        "dataset_summary": dataset_summary,
        "dataset_name": get_dataset_name(),
        "rows": df.shape[0],
        "columns": df.shape[1],
        "health_score": health["score"],
        "health_grade": health["grade"],
        "health_summary": health["summary"],
        "health_breakdown": health["breakdown"],
        "issues": [f"{i['title']}: {i['description']}" for i in issues[:10]],
        "problem_type": st.session_state.get(SessionKeys.PROBLEM_TYPE) or "Not trained",
        "best_model": st.session_state.get(SessionKeys.BEST_MODEL_NAME) or "Not trained",
        "target_column": st.session_state.get(SessionKeys.TARGET_COLUMN) or "N/A",
        "confidence": confidence,
        "model_results": model_lines,
        "ai_insights": insights,
        "shap_ranking": shap_lines,
        "eda_findings": eda_insights,
        "cleaning_actions": st.session_state.get(SessionKeys.CLEANING_REPORT) or ["No cleaning performed"],
        "recommendations": recs,
        "conclusion": (
            f"Analysis complete. Best model: {st.session_state.get(SessionKeys.BEST_MODEL_NAME, 'N/A')} "
            f"with confidence {confidence}%. Recommended for further validation and deployment."
        ),
    }

    path = generate_pdf_report(payload)
    st.session_state[SessionKeys.REPORT_PAYLOAD] = payload
    st.session_state[SessionKeys.REPORT_PATH] = path
    st.session_state[SessionKeys.REPORT_GENERATED] = True
    st.session_state[SessionKeys.AI_INSIGHTS] = insights
    st.session_state[SessionKeys.RECOMMENDATIONS] = recs
    st.session_state[SessionKeys.CONFIDENCE_SCORE] = confidence
