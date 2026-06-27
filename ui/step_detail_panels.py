"""Rich expandable panels for AI Command Center pipeline stages."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agents.eda_agent import correlation_heatmap, missing_values_chart
from config import ACCENT_COLOR, DANGER_COLOR, PRIMARY_COLOR, SUCCESS_COLOR, WARNING_COLOR
from ui.chief_decision_panel import render_chief_decision_panel
from ui.interactive_report_center import render_interactive_report_center
from ui.components import glass_panel_small, require_dataset
from utils.error_boundary import log_exception, render_panel_error
from utils.health_score import compute_health_score, detect_data_issues
from utils.safe_checks import (
    coalesce_dict,
    coalesce_list,
    format_score_display,
    coerce_numeric_score,
    is_present,
    normalize_feature_importance,
    normalize_recommendations,
    safe_dict_get,
)
from utils.session_manager import SessionKeys, get_dataframe, get_dataset_name, get_problem_type, has_dataset

_STAGE_OUTPUT_FALLBACK = {
    "dataset_intelligence": ("dataset_report", "dataset_analysis"),
    "data_cleaning": ("cleaning_results",),
    "eda": ("eda_results",),
    "feature_engineering": ("feature_engineering_results",),
    "automl": ("model_results",),
    "model_comparison": ("model_comparison",),
    "explainability": ("explainability_results", "xai_results"),
    "ai_ethics_trust": ("ai_trust_results", "ethics_report"),
    "self_improvement": ("improvement_history",),
    "deployment_readiness": ("deployment_readiness",),
    "ai_decision": ("recommendation",),
    "pdf_report": ("final_report",),
    "hyperparameter_optimization": ("hyperparameter_report",),
}


def _render_ethics_assessment(ethics: Dict[str, Any], render_stat_grid: Optional[Callable] = None) -> None:
    """Render trust & fairness results as executive cards — never raw JSON keys."""
    trust_score = ethics.get("trust_score")
    fairness_score = ethics.get("fairness_score")
    governance = coalesce_dict(safe_dict_get(ethics, "ai_governance_score"))
    if trust_score is None and governance:
        trust_score = governance.get("score")
    if fairness_score is None:
        fairness_score = coalesce_dict(safe_dict_get(ethics, "fairness")).get("fairness_score")

    stats = [
        {"label": "Overall Trust Score", "value": format_score_display(trust_score, suffix="%", unavailable="—")},
        {"label": "Fairness Score", "value": format_score_display(fairness_score, suffix="%", unavailable="Not Evaluated")},
        {"label": "Bias Risk", "value": safe_dict_get(ethics, "bias_risk") or safe_dict_get(ethics, "risk_level") or "—"},
        {"label": "Privacy Assessment", "value": "Reviewed" if normalize_recommendations(safe_dict_get(ethics, "privacy_concerns")) else "—"},
    ]
    if render_stat_grid:
        render_stat_grid(stats, columns=2)
    else:
        cols = st.columns(2)
        for idx, item in enumerate(stats):
            with cols[idx % 2]:
                st.metric(item["label"], item["value"])

    summary = safe_dict_get(ethics, "executive_summary")
    if summary:
        st.markdown(
            f"<div class='report-insight-box'><strong>Executive Summary</strong><br>{summary}</div>",
            unsafe_allow_html=True,
        )

    bias = normalize_recommendations(safe_dict_get(ethics, "bias_concerns"))
    if not bias:
        bias = normalize_recommendations(safe_dict_get(coalesce_dict(safe_dict_get(ethics, "bias_analysis")), "bias_concerns"))
    if not bias:
        bias = normalize_recommendations(safe_dict_get(ethics, "concerns"))

    fairness = normalize_recommendations(safe_dict_get(ethics, "fairness_concerns"))
    privacy = normalize_recommendations(safe_dict_get(ethics, "privacy_concerns"))
    if not privacy:
        privacy = normalize_recommendations(safe_dict_get(coalesce_dict(safe_dict_get(ethics, "privacy_analysis")), "detected_identifiers"))

    compliance = normalize_recommendations(safe_dict_get(ethics, "compliance_recommendations"))
    if not compliance:
        compliance = normalize_recommendations(safe_dict_get(ethics, "recommendations"))

    if bias:
        st.markdown("**Bias Concerns**")
        for item in bias[:8]:
            st.markdown(f"- {item}")
    if fairness:
        st.markdown("**Fairness Concerns**")
        for item in fairness[:8]:
            st.markdown(f"- {item}")
    if privacy:
        st.markdown("**Privacy Risks**")
        for item in privacy[:8]:
            st.markdown(f"- {item}")
    if compliance:
        st.markdown("**Compliance Recommendations**")
        for item in compliance[:8]:
            st.markdown(f"- {item}")


def _render_monitoring_panel(output: Optional[Dict[str, Any]], result: Any) -> None:
    """Always show monitoring guidance — never a blank panel."""
    st.markdown("**Monitoring Status:** Ready")
    st.markdown("**Retraining Trigger:** Performance drift > 5%")
    st.markdown("**Suggested Monitoring Metrics:**")
    for metric in ("Accuracy", "F1", "Data Drift", "Concept Drift"):
        st.markdown(f"- {metric}")

    drift = coalesce_dict(safe_dict_get(output, "drift_report"))
    if not drift and isinstance(result, dict):
        drift = result
    if drift:
        severity = safe_dict_get(drift, "severity") or safe_dict_get(drift, "drift_level")
        if severity:
            st.markdown(f"**Latest drift severity:** {severity}")
        summary = safe_dict_get(drift, "summary") or safe_dict_get(drift, "message")
        if summary:
            st.write(summary)


def _render_prediction_workspace(output: Optional[Dict[str, Any]]) -> None:
    """Sample prediction workspace with editable rows."""
    model = st.session_state.get(SessionKeys.BEST_MODEL)
    X = st.session_state.get(SessionKeys.X_DATA)
    best_name = st.session_state.get(SessionKeys.BEST_MODEL_NAME) or "Model"
    problem_type = get_problem_type(output)
    trained = st.session_state.get(SessionKeys.MODEL_TRAINED, False)
    if not trained and output:
        trained = bool(safe_dict_get(coalesce_dict(safe_dict_get(output, "model_results")), "best_model"))

    if not trained or model is None or not is_present(X):
        st.info("Train a model first.")
        return

    st.markdown("### Sample Prediction Workspace")
    st.caption(f"Using trained model: **{best_name}** ({problem_type})")
    sample = X.head(5).copy()
    edited = st.data_editor(sample, num_rows="fixed", width="stretch", key="prediction_sample_editor")

    if st.button("Run Predictions", type="primary", key="prediction_sample_run"):
        try:
            predictions = model.predict(edited)
            result_df = edited.copy()
            result_df["Prediction"] = predictions
            if hasattr(model, "predict_proba") and str(problem_type).lower() == "classification":
                probas = model.predict_proba(edited)
                result_df["Confidence"] = [f"{max(row) * 100:.1f}%" for row in probas]
            st.success("Predictions generated successfully.")
            st.dataframe(result_df, width="stretch")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def resolve_stage_result(
    stage_results: Dict[str, Any],
    ui_step_key: str,
    output: Optional[Dict[str, Any]],
    backend_key: str,
) -> Any:
    """Resolve stage data from live results or completed pipeline output."""
    if stage_results.get(backend_key) is not None:
        return stage_results.get(backend_key)
    if not output:
        return None
    for fallback_key in _STAGE_OUTPUT_FALLBACK.get(ui_step_key, ()):
        value = output.get(fallback_key)
        if value is not None:
            return value
    return None


def render_step_detail(
    step_key: str,
    output: Optional[Dict[str, Any]],
    stage_results: Dict[str, Any],
    *,
    backend_key: str,
    open_panel: Callable[[str, Optional[str]], None],
    close_panel: Callable[[], None],
    render_stat_grid: Callable[[List[Dict[str, Any]], int], None],
    render_stage_output: Callable[[Any, str], None],
) -> None:
    """Render a single pipeline stage detail panel."""
    try:
        _render_step_detail_impl(
            step_key,
            output,
            stage_results,
            backend_key=backend_key,
            open_panel=open_panel,
            close_panel=close_panel,
            render_stat_grid=render_stat_grid,
            render_stage_output=render_stage_output,
        )
    except Exception as exc:
        log_exception(step_key, exc)
        render_panel_error(step_key.replace("_", " ").title(), exc)
        close_panel()


def _render_step_detail_impl(
    step_key: str,
    output: Optional[Dict[str, Any]],
    stage_results: Dict[str, Any],
    *,
    backend_key: str,
    open_panel: Callable[[str, Optional[str]], None],
    close_panel: Callable[[], None],
    render_stat_grid: Callable[[List[Dict[str, Any]], int], None],
    render_stage_output: Callable[[Any, str], None],
) -> None:
    """Inner implementation for stage detail rendering."""
    result = resolve_stage_result(stage_results, step_key, output, backend_key)

    if step_key == "dataset_upload":
        open_panel("Dataset Upload", "Confirm your file and dataset metadata.")
        if has_dataset():
            df = get_dataframe()
            health = compute_health_score(df)
            issues = detect_data_issues(df)
            st.write(f"**Dataset:** {get_dataset_name()}")
            st.write(f"**Rows:** {df.shape[0]:,} · **Columns:** {df.shape[1]}")
            st.write(f"**Missing values:** {int(df.isnull().sum().sum()):,}")
            st.write(f"**Duplicates:** {int(df.duplicated().sum()):,}")
            render_stat_grid(
                [
                    {"label": "Health Score", "value": f"{health['score']:.0f}/100"},
                    {"label": "Grade", "value": health.get("grade", "N/A")},
                    {"label": "Issues Found", "value": len(issues)},
                ],
                columns=3,
            )
            if issues:
                st.write("**Detected Issues:**")
                for issue in issues[:5]:
                    st.write(f"- {issue}")
        else:
            require_dataset()
        close_panel()
        return

    if step_key == "dataset_intelligence":
        open_panel("Dataset Intelligence", "AI-driven dataset analysis and insights.")
        if isinstance(result, dict):
            meta = coalesce_dict(st.session_state.get(SessionKeys.DATASET_METADATA))
            result = safe_dict_get(meta, "dataset_report") or safe_dict_get(meta, "intelligence_report")
        if isinstance(result, dict):
            problem = coalesce_dict(safe_dict_get(result, "problem_analysis"))
            summary = safe_dict_get(problem, "summary") or safe_dict_get(result, "summary")
            if is_present(summary):
                if isinstance(summary, str):
                    st.write(summary)
                else:
                    render_stage_output(summary, "Dataset Summary")
            problem_type = safe_dict_get(problem, "problem_type") or safe_dict_get(result, "problem_type", "Unknown")
            target = safe_dict_get(problem, "likely_target") or safe_dict_get(result, "target", "Unknown")
            score_block = coalesce_dict(safe_dict_get(result, "intelligence_score"))
            intelligence_score = score_block.get("score", "N/A")
            if has_dataset():
                df = get_dataframe()
                dtypes = df.dtypes.astype(str).value_counts().to_dict()
                missing_total = int(df.isnull().sum().sum())
                render_stat_grid(
                    [
                        {"label": "Rows", "value": f"{df.shape[0]:,}"},
                        {"label": "Columns", "value": df.shape[1]},
                        {"label": "Missing Values", "value": missing_total},
                        {"label": "Duplicates", "value": int(df.duplicated().sum())},
                        {"label": "Problem Type", "value": problem_type},
                        {"label": "Target Column", "value": target},
                        {"label": "Intelligence Score", "value": f"{intelligence_score}/100"},
                    ],
                    columns=3,
                )
                st.write("**Feature Types:**")
                st.write(", ".join(f"{k}: {v}" for k, v in dtypes.items()))
                if missing_total:
                    st.plotly_chart(missing_values_chart(df), width="stretch")
            else:
                render_stat_grid(
                    [
                        {"label": "Problem Type", "value": problem_type},
                        {"label": "Target Column", "value": target},
                        {"label": "Intelligence Score", "value": f"{intelligence_score}/100"},
                    ],
                    columns=3,
                )
        elif result is not None:
            render_stage_output(result, "Dataset intelligence result")
        else:
            st.info("Run the autonomous pipeline to generate dataset intelligence.")
        close_panel()
        return

    if step_key == "data_cleaning":
        open_panel("Data Cleaning", "Inspect cleaned data and improvement actions.")
        if isinstance(result, dict):
            report = result.get("report", {})
            shape = result.get("shape", (0, 0))
            before_shape = result.get("before_shape") or result.get("original_shape")
            st.write(f"**Cleaned dataset shape:** {shape[0]:,} rows × {shape[1]} columns")
            if before_shape:
                st.write(f"**Before cleaning:** {before_shape[0]:,} rows × {before_shape[1]} columns")
            if isinstance(report, dict):
                st.write(f"**Status:** {report.get('status', 'unknown')}")
                actions = coalesce_list(safe_dict_get(report, "actions"))
                if not actions:
                    actions = coalesce_list(safe_dict_get(report, "steps"))
                if actions:
                    st.write("**Cleaning actions:**")
                    for action in actions:
                        st.write(f"- {action}")
            else:
                render_stage_output(report, "Cleaning report")
        elif result is not None:
            render_stage_output(result, "Cleaning result")
        else:
            st.info("No cleaning results yet.")
        close_panel()
        return

    if step_key == "eda":
        open_panel("EDA Exploration", "Dataset visualizations and insights.")
        if isinstance(result, dict):
            summary = result.get("summary")
            if is_present(summary):
                render_stage_output(summary, "Statistical Summary")
            numerics = coalesce_list(safe_dict_get(result, "numerical_columns"))
            categoricals = coalesce_list(safe_dict_get(result, "categorical_columns"))
            render_stat_grid(
                [
                    {"label": "Numeric Features", "value": len(numerics)},
                    {"label": "Categorical Features", "value": len(categoricals)},
                ],
                columns=2,
            )
            if has_dataset():
                df = get_dataframe()
                st.plotly_chart(correlation_heatmap(df), width="stretch")
                if numerics:
                    fig = px.histogram(
                        df,
                        x=numerics[0],
                        nbins=30,
                        title=f"Distribution: {numerics[0]}",
                        color_discrete_sequence=[ACCENT_COLOR],
                    )
                    fig.update_layout(template="plotly_white", height=420)
                    st.plotly_chart(fig, width="stretch")
            insights = coalesce_list(safe_dict_get(result, "insights"))
            if not insights:
                insights = coalesce_list(safe_dict_get(result, "ai_insights"))
            if is_present(insights):
                st.write("**AI Insights:**")
                for insight in insights[:8]:
                    st.write(f"- {insight}")
            charts = result.get("charts")
            if is_present(charts):
                render_stage_output(charts, "Charts")
            if not is_present(summary) and not is_present(insights) and not is_present(charts):
                render_stage_output(result, "EDA result")
        elif result is not None:
            render_stage_output(result, "EDA result")
        else:
            st.info("No EDA results yet.")
        close_panel()
        return

    if step_key == "feature_engineering":
        open_panel("Feature Engineering", "Generated features and transformations.")
        if isinstance(result, dict):
            created = coalesce_list(safe_dict_get(result, "created_features"))
            if not created:
                created = coalesce_list(safe_dict_get(result, "new_features"))
            selected = coalesce_list(safe_dict_get(result, "selected_features"))
            if not selected:
                selected = coalesce_list(safe_dict_get(result, "features"))
            fi_records = normalize_feature_importance(safe_dict_get(result, "feature_importance"))
            steps = coalesce_list(safe_dict_get(result, "steps"))
            if not steps:
                steps = coalesce_list(safe_dict_get(result, "transformations"))
            recommendations = normalize_recommendations(safe_dict_get(result, "recommended_changes"))
            if not recommendations:
                recommendations = normalize_recommendations(safe_dict_get(result, "recommendations"))
            if steps:
                st.write("**Engineering Steps:**")
                for step in steps:
                    st.write(f"- {step}")
            if is_present(created):
                st.write("**Created Features:**")
                for feat in created[:10]:
                    st.write(f"- {feat}")
            if is_present(selected):
                st.write("**Selected Features:**")
                st.write(", ".join(str(f) for f in selected[:15]))
            if fi_records:
                ranked = sorted(fi_records, key=lambda x: abs(x["importance"]), reverse=True)[:10]
                fig = go.Figure(
                    go.Bar(
                        x=[item["importance"] for item in ranked],
                        y=[item["feature"] for item in ranked],
                        orientation="h",
                        marker_color=PRIMARY_COLOR,
                    )
                )
                fig.update_layout(title="Feature Importance", template="plotly_white", height=360)
                st.plotly_chart(fig, width="stretch")
            if recommendations:
                st.write("**Recommendations:**")
                for rec in recommendations[:5]:
                    st.write(f"- {rec}")
            if not steps and not is_present(created) and not recommendations:
                render_stage_output(result, "Feature engineering result")
        elif result is not None:
            render_stage_output(result, "Feature engineering result")
        else:
            st.info("No feature engineering results yet.")
        close_panel()
        return

    if step_key == "automl":
        open_panel("AutoML Training", "Model leaderboard and training results.")
        model_data = result if isinstance(result, dict) else coalesce_dict(safe_dict_get(output, "model_results"))
        if isinstance(model_data, dict) and model_data.get("metrics"):
            metrics = model_data["metrics"]
            best_model = model_data.get("best_model")
            training_times = coalesce_dict(safe_dict_get(model_data, "training_times"))
            rows = []
            for rank, (name, score) in enumerate(
                sorted(metrics.items(), key=lambda x: x[1], reverse=True),
                start=1,
            ):
                pct = f"{score * 100:.1f}%" if score <= 1 else f"{score:.4f}"
                rows.append(
                    {
                        "Rank": rank,
                        "Model": name,
                        "Score": pct,
                        "Training Time": training_times.get(name, "—"),
                        "Status": "✓ Best" if name == best_model else "",
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            top_score = max(metrics.values()) if metrics else 0
            top_display = f"{top_score * 100:.1f}%" if top_score <= 1 else f"{top_score:.4f}"
            glass_panel_small(
                f"<strong>Best Model:</strong> {best_model or 'TBD'}<br/>"
                f"<strong>Top Score:</strong> {top_display}<br/>"
                f"<strong>Why selected:</strong> Highest validation score among evaluated candidates."
            )
            if len(metrics) > 1:
                fig = go.Figure(go.Bar(x=list(metrics.keys()), y=list(metrics.values()), marker_color=ACCENT_COLOR))
                fig.update_layout(title="Model Comparison", template="plotly_white", height=360)
                st.plotly_chart(fig, width="stretch")
        elif model_data is not None:
            render_stage_output(model_data, "AutoML result")
        else:
            st.info("No model training results yet.")
        close_panel()
        return

    if step_key == "model_comparison":
        open_panel("Model Comparison", "Compare candidate models and optimization outcomes.")
        if result is not None:
            render_stage_output(result, "Model comparison result")
        else:
            st.info("No model comparison results yet.")
        close_panel()
        return

    if step_key == "explainability":
        open_panel("Explainability (SHAP)", "Feature importance and model interpretation.")
        xai = result if isinstance(result, dict) else coalesce_dict(safe_dict_get(output, "explainability_results"))
        if not xai:
            xai = coalesce_dict(safe_dict_get(output, "xai_results"))
        if isinstance(xai, dict):
            fi_raw = safe_dict_get(xai, "feature_importance")
            if not is_present(fi_raw):
                fi_raw = st.session_state.get(SessionKeys.SHAP_IMPORTANCE)
            if not is_present(fi_raw):
                fi_raw = st.session_state.get(SessionKeys.FEATURE_IMPORTANCE)
            if not is_present(fi_raw):
                fi_raw = safe_dict_get(xai, "shap_values")
            fi_records = normalize_feature_importance(fi_raw)
            explanation = safe_dict_get(xai, "explanation") or safe_dict_get(xai, "summary") or safe_dict_get(xai, "natural_language")
            if fi_records:
                ranked = sorted(fi_records, key=lambda x: abs(x["importance"]), reverse=True)[:12]
                st.write("**Top Features by Importance:**")
                for item in ranked[:8]:
                    importance = item["importance"]
                    direction = "positive impact" if importance >= 0 else "negative impact"
                    st.write(f"- **{item['feature']}**: {importance:.4f} ({direction})")
                fig = go.Figure(
                    go.Bar(
                        x=[abs(item["importance"]) for item in ranked],
                        y=[item["feature"] for item in ranked],
                        orientation="h",
                        marker_color=[
                            SUCCESS_COLOR if item["importance"] >= 0 else DANGER_COLOR for item in ranked
                        ],
                    )
                )
                fig.update_layout(title="SHAP Feature Importance", template="plotly_white", height=420)
                st.plotly_chart(fig, width="stretch")
            elif is_present(fi_raw):
                render_stage_output(fi_raw, "Feature importance")
            if explanation:
                st.write("**Natural Language Explanation:**")
                st.write(explanation)
            if not fi_records and not explanation:
                if st.session_state.get(SessionKeys.SHAP_COMPUTED):
                    st.warning("SHAP completed, but no chartable feature importance was found in session or pipeline output.")
                else:
                    render_stage_output(xai, "Explainability result")
        elif xai is not None:
            render_stage_output(xai, "Explainability result")
        else:
            st.info("No explainability results yet.")
        close_panel()
        return

    if step_key == "ai_ethics_trust":
        open_panel("AI Ethics & Trust", "Fairness, bias, and trust assessment.")
        executive_metrics = st.session_state.get(SessionKeys.EXECUTIVE_METRICS) or {}
        ethics = result if isinstance(result, dict) else coalesce_dict(safe_dict_get(output, "ai_trust_results"))
        if not ethics and executive_metrics:
            ethics = coalesce_dict(executive_metrics)
        if not ethics:
            ethics = coalesce_dict(safe_dict_get(output, "ethics_report"))
        if isinstance(ethics, dict) and ethics:
            _render_ethics_assessment(ethics, render_stat_grid=render_stat_grid)
        elif ethics is not None:
            render_stage_output(ethics, "Ethics result")
        else:
            st.info("No ethics assessment yet.")
        close_panel()
        return

    if step_key == "self_improvement":
        open_panel("Self-Improvement", "Model optimization and iterative improvements.")
        if isinstance(result, dict):
            history = coalesce_list(safe_dict_get(result, "improvement_history"))
            if history:
                st.write("**Improvement History:**")
                for event in history[:8]:
                    st.write(f"- {event}")
            else:
                render_stage_output(result, "Self improvement result")
        elif result is not None:
            render_stage_output(result, "Self improvement result")
        else:
            st.info("No self-improvement results yet.")
        close_panel()
        return

    if step_key == "deployment_readiness":
        open_panel("Deployment Readiness", "Risk assessment and deployment metrics.")
        executive_metrics = st.session_state.get(SessionKeys.EXECUTIVE_METRICS) or {}
        deploy = result if isinstance(result, dict) else coalesce_dict(safe_dict_get(output, "deployment_readiness"))
        if not deploy and executive_metrics:
            deploy = coalesce_dict(safe_dict_get(executive_metrics, "deployment_readiness")) or coalesce_dict(executive_metrics.get("deployment_decision") or {})
        if isinstance(deploy, dict):
            risk_level = deploy.get("risk_level", "Unknown") or safe_dict_get(executive_metrics, "risk_level", "Unknown")
            readiness_score = (
                deploy.get("readiness_score")
                or deploy.get("deployment_score")
                or safe_dict_get(executive_metrics, "confidence_score")
                or safe_dict_get(output, "final_score", 0)
            )
            risk_color = {"Low": SUCCESS_COLOR, "Medium": WARNING_COLOR, "High": DANGER_COLOR}.get(risk_level, PRIMARY_COLOR)
            render_stat_grid(
                [
                    {"label": "Production Readiness", "value": f"{readiness_score}/100"},
                    {"label": "Risk Level", "value": risk_level},
                ],
                columns=2,
            )
            st.markdown(
                f"<div style='padding:1rem;background:rgba(99,102,241,0.08);border-radius:12px;border-left:4px solid {risk_color};'>"
                f"<strong>Risk Assessment:</strong> <span style='color:{risk_color};'>{risk_level}</span></div>",
                unsafe_allow_html=True,
            )
            warnings = normalize_recommendations(safe_dict_get(deploy, "warnings"))
            recommendations = normalize_recommendations(safe_dict_get(deploy, "recommendations"))
            if not recommendations:
                recommendations = normalize_recommendations(safe_dict_get(deploy, "api_recommendations"))
            if warnings:
                st.write("**Warnings:**")
                for warning in warnings[:5]:
                    st.write(f"⚠️ {warning}")
            if recommendations:
                st.write("**API Deployment Recommendations:**")
                for rec in recommendations[:5]:
                    st.write(f"✓ {rec}")
            if not warnings and not recommendations:
                render_stage_output(deploy, "Deployment readiness result")
        elif deploy is not None:
            render_stage_output(deploy, "Deployment readiness result")
        else:
            st.info("No deployment readiness assessment yet.")
        close_panel()
        return

    if step_key == "ai_decision":
        open_panel("Final AI Decision", "Executive AI Chief Data Scientist recommendation.")
        render_chief_decision_panel(output)
        if isinstance(result, dict):
            with st.expander("Raw decision payload", expanded=False):
                render_stage_output(result, "AI decision result")
        close_panel()
        return

    if step_key == "prediction":
        open_panel("Prediction Engine", "Sample predictions with selected model.")
        _render_prediction_workspace(output)
        close_panel()
        return

    if step_key == "pdf_report":
        open_panel("Interactive AI Report", "Browser-first executive report with optional PDF export.")
        render_interactive_report_center(output)
        close_panel()
        return

    if step_key == "monitoring":
        open_panel("Monitoring & Drift", "Production monitoring guidance.")
        _render_monitoring_panel(output, result)
        close_panel()
        return

    if step_key == "hyperparameter_optimization":
        open_panel("Hyperparameter Optimization", "Tuning strategy and recommended parameters.")
        hpo = result if isinstance(result, dict) else coalesce_dict(safe_dict_get(output, "hyperparameter_report"))
        if hpo is not None:
            render_stage_output(hpo, "Hyperparameter report")
        else:
            st.info("No hyperparameter report yet.")
        close_panel()
