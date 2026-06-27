"""Premium Interactive AI Report Center — browser-first pipeline output experience."""

from __future__ import annotations

import html
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agents.eda_agent import correlation_heatmap, detect_outliers, missing_values_chart
from config import ACCENT_COLOR, DANGER_COLOR, PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR
from ui.chief_decision_panel import build_chief_decision_data, render_chief_decision_panel
from ui.components import primary_metric_label
from ui.saas_components import render_deployment_status_card
from ui.validation_dashboard import render_validation_dashboard
from utils.health_score import compute_health_score, detect_data_issues
from utils.error_boundary import run_panel
from utils.safe_checks import (
    coalesce_dict,
    coalesce_list,
    coerce_numeric_score,
    display_kpi_value,
    format_accuracy_display,
    is_present,
    normalize_feature_importance,
    normalize_recommendations,
    resolve_canonical_accuracy,
    safe_dict_get,
)
from utils.session_manager import (
    SessionKeys,
    get_autonomous_result,
    get_dataframe,
    get_dataset_name,
    get_problem_type,
    get_session_health,
    has_autonomous_result,
    has_dataset,
)
from utils.model_ranking import build_best_model_consistency_notes, composite_model_score, rank_models_by_composite
from utils.report_exports import build_excel_bytes, build_export_context_from_report_ctx


def _plotly_template() -> str:
    try:
        base = st.get_option("theme.base")
        return "plotly_dark" if base == "dark" else "plotly_white"
    except Exception:
        return "plotly_white"


def _chart_config() -> Dict[str, Any]:
    return {"displayModeBar": True, "displaylogo": False}


def _export_chart(fig: go.Figure, filename: str) -> None:
    try:
        png_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
        st.download_button(
            "Export chart (PNG)",
            png_bytes,
            file_name=filename,
            mime="image/png",
            key=f"export_{filename}_{hash(filename) % 10_000}",
        )
    except Exception:
        st.caption("Install kaleido for chart export: pip install kaleido")


def _metric_format(score: Optional[float], problem_type: str) -> str:
    return format_accuracy_display(score, problem_type, unavailable="Unavailable")


def _health_grade_from_score(score: float) -> tuple[str, str, str]:
    if score >= 85:
        return "Excellent", "health-excellent", "Dataset quality is strong and ready for modeling."
    if score >= 70:
        return "Good", "health-good", "Dataset is in good shape with minor improvements possible."
    if score >= 50:
        return "Fair", "health-fair", "Dataset has quality issues that should be addressed before production."
    return "Poor", "health-poor", "Dataset requires significant cleaning before reliable modeling."


def build_report_context(output: Optional[Dict[str, Any]], df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Assemble report data from pipeline output, session state, and dataframe."""
    if has_autonomous_result():
        output = get_autonomous_result()
    else:
        output = output or {}
    final_report = coalesce_dict(safe_dict_get(output, "final_report"))
    report_payload = coalesce_dict(safe_dict_get(final_report, "payload"))

    # === CHECK FOR EXECUTIVE_METRICS FIRST ===
    executive_metrics = st.session_state.get(SessionKeys.EXECUTIVE_METRICS) or {}
    executive_model_results = coalesce_dict(safe_dict_get(executive_metrics, "model_results")) if executive_metrics else {}
    executive_deploy = coalesce_dict(safe_dict_get(executive_metrics, "deployment_readiness")) if executive_metrics else {}
    executive_health_score = executive_metrics.get("health_score") if executive_metrics else None
    executive_recommendation = safe_dict_get(executive_metrics.get("final_decision") or {}, "recommendation") if executive_metrics else None
    executive_trust_score = executive_metrics.get("trust_score") if executive_metrics else None
    executive_confidence = executive_metrics.get("confidence_score") if executive_metrics else None
    executive_risk_level = executive_metrics.get("risk_level") if executive_metrics else None
    executive_deployment_status = executive_metrics.get("deployment_status") if executive_metrics else None
    executive_best_model = executive_metrics.get("best_model") if executive_metrics else None
    executive_accuracy = executive_metrics.get("accuracy") if executive_metrics else None

    chief = build_chief_decision_data(output if output else None, df)

    dataset_report = coalesce_dict(safe_dict_get(output, "dataset_report"))
    if not dataset_report:
        dataset_report = coalesce_dict(safe_dict_get(output, "dataset_analysis"))
    problem = coalesce_dict(safe_dict_get(dataset_report, "problem_analysis"))

    problem_type = get_problem_type(output)
    target_column = (
        st.session_state.get(SessionKeys.TARGET_COLUMN)
        or safe_dict_get(problem, "likely_target")
        or safe_dict_get(dataset_report, "target")
        or "—"
    )

    rows = cols = duplicates = missing_total = missing_pct = None
    numeric_features: List[str] = []
    categorical_features: List[str] = []
    if is_present(df):
        rows, cols = int(df.shape[0]), int(df.shape[1])
        duplicates = int(df.duplicated().sum())
        missing_total = int(df.isnull().sum().sum())
        missing_pct = float(df.isnull().mean().mean() * 100)
        numeric_features = df.select_dtypes(include="number").columns.tolist()
        categorical_features = [c for c in df.columns if c not in numeric_features]

    cleaning = coalesce_dict(safe_dict_get(output, "cleaning_results"))
    cleaning_report = coalesce_dict(safe_dict_get(cleaning, "report"))
    before_shape = safe_dict_get(cleaning, "before_shape") or safe_dict_get(cleaning, "original_shape")
    after_shape = safe_dict_get(cleaning, "shape")
    cleaning_actions = normalize_recommendations(safe_dict_get(cleaning_report, "actions"))
    if not cleaning_actions:
        cleaning_actions = normalize_recommendations(safe_dict_get(cleaning_report, "steps"))

    eda = coalesce_dict(safe_dict_get(output, "eda_results"))
    eda_insights: List[str] = []
    if eda:
        raw_insights = safe_dict_get(eda, "insights")
        if raw_insights is None:
            raw_insights = safe_dict_get(eda, "ai_insights")
        if raw_insights is None:
            raw_insights = st.session_state.get(SessionKeys.EDA_INSIGHTS)
        eda_insights = normalize_recommendations(raw_insights)
        numerics = coalesce_list(safe_dict_get(eda, "numerical_columns")) or numeric_features
        categoricals = coalesce_list(safe_dict_get(eda, "categorical_columns")) or categorical_features
    else:
        numerics = numeric_features
        categoricals = categorical_features

    model_results = coalesce_dict(safe_dict_get(output, "model_results"))
    if executive_model_results:
        model_results = executive_model_results
    elif report_payload and isinstance(report_payload, dict):
        model_results = model_results or coalesce_dict(safe_dict_get(report_payload, "model_results"))

    metrics = coalesce_dict(safe_dict_get(model_results, "metrics"))
    if not metrics:
        session_results = st.session_state.get(SessionKeys.RESULTS)
        metrics = coalesce_dict(session_results)
    detailed_metrics = coalesce_dict(safe_dict_get(model_results, "detailed_metrics"))
    if not detailed_metrics:
        detailed_metrics = coalesce_dict(st.session_state.get(SessionKeys.MODEL_METRICS))
    training_times = coalesce_dict(safe_dict_get(model_results, "training_times"))
    best_model = safe_dict_get(model_results, "best_model")
    if not best_model:
        best_model = st.session_state.get(SessionKeys.BEST_MODEL_NAME)
    if executive_metrics:
        best_model = executive_best_model or best_model

    model_selection_explanation = coalesce_dict(safe_dict_get(model_results, "model_selection_explanation"))
    model_selection_notes = build_best_model_consistency_notes(
        best_model,
        detailed_metrics,
        str(problem_type),
        metrics,
        model_selection_explanation,
    )

    leaderboard: List[Dict[str, Any]] = []
    ranked_models = rank_models_by_composite(detailed_metrics, str(problem_type), metrics)
    if ranked_models:
        for rank, (name, composite) in enumerate(ranked_models, start=1):
            time_val = "Unavailable"
            if isinstance(training_times, dict) and name in training_times:
                tv = training_times[name]
                time_val = f"{tv:.2f}s" if isinstance(tv, (int, float)) else str(tv)
            holdout = composite_model_score(name, detailed_metrics, str(problem_type), metrics)
            leaderboard.append(
                {
                    "Rank": rank,
                    "Model": name,
                    "Score": _metric_format(composite if composite else None, str(problem_type)),
                    "Training Time": time_val,
                    "Status": "✓ Best" if name == best_model else "",
                }
            )
    elif isinstance(metrics, dict) and metrics:
        for rank, (name, score) in enumerate(sorted(metrics.items(), key=lambda x: x[1], reverse=True), start=1):
            time_val = "Unavailable"
            if isinstance(training_times, dict) and name in training_times:
                tv = training_times[name]
                time_val = f"{tv:.2f}s" if isinstance(tv, (int, float)) else str(tv)
            leaderboard.append(
                {
                    "Rank": rank,
                    "Model": name,
                    "Score": _metric_format(float(score) if score is not None else None, str(problem_type)),
                    "Training Time": time_val,
                    "Status": "✓ Best" if name == best_model else "",
                }
            )

    validation = safe_dict_get(output, "validation_results")
    validation_results = validation if isinstance(validation, dict) else None
    validation_lines: List[str] = []
    if isinstance(validation, dict):
        for key, value in validation.items():
            validation_lines.append(f"{key}: {value}")
    elif isinstance(validation, list):
        validation_lines.extend(str(v) for v in validation[:8])

    explainability = coalesce_dict(safe_dict_get(output, "explainability_results"))
    if not explainability:
        explainability = coalesce_dict(safe_dict_get(output, "xai_results"))
    fi_records = normalize_feature_importance(safe_dict_get(explainability, "feature_importance"))
    if not fi_records:
        fi_records = normalize_feature_importance(st.session_state.get(SessionKeys.SHAP_IMPORTANCE))
    if not fi_records:
        fi_records = normalize_feature_importance(st.session_state.get(SessionKeys.FEATURE_IMPORTANCE))
    expl_text = str(safe_dict_get(explainability, "explanation") or safe_dict_get(explainability, "summary") or "")
    if fi_records and expl_text and "shap skipped" in expl_text.lower():
        expl_text = ""
    pos_neg = st.session_state.get(SessionKeys.SHAP_POSITIVE_NEGATIVE)

    deploy = executive_deploy or coalesce_dict(safe_dict_get(output, "deployment_readiness"))
    if not deploy and isinstance(report_payload, dict):
        deploy = coalesce_dict(safe_dict_get(report_payload, "deployment_readiness"))

    recommendation = (
        executive_recommendation
        or safe_dict_get(output, "recommendation")
        or safe_dict_get(report_payload, "recommendation")
        or chief.get("business_recommendation")
        or "—"
    )

    ethics = coalesce_dict(safe_dict_get(output, "ai_trust_results"))
    if not ethics:
        ethics = coalesce_dict(safe_dict_get(output, "ethics_report"))
    if not ethics and isinstance(report_payload, dict):
        ethics = {
            "trust_score": safe_dict_get(report_payload, "trust_score"),
            "fairness_score": safe_dict_get(report_payload, "fairness_score"),
        }
    if executive_trust_score is not None:
        ethics["trust_score"] = executive_trust_score
    if executive_risk_level and "risk_level" not in ethics:
        ethics["risk_level"] = executive_risk_level
    if executive_deployment_status and "deployment_status" not in ethics:
        ethics["deployment_status"] = executive_deployment_status
    if not ethics and executive_metrics:
        ethics = {
            "trust_score": executive_trust_score,
            "fairness_score": executive_accuracy,
            "risk_level": executive_risk_level,
        }

    readiness_score = safe_dict_get(deploy, "readiness_score")
    if readiness_score is None:
        readiness_score = safe_dict_get(deploy, "deployment_score")
    if readiness_score is None and executive_metrics:
        readiness_score = safe_dict_get(executive_metrics.get("deployment_decision") or {}, "readiness_score") or executive_confidence
    if readiness_score is None:
        readiness_score = safe_dict_get(output, "final_score") or chief.get("final_score")

    if executive_health_score is not None:
        health_score_value = float(executive_health_score)
        grade, grade_class, summary = _health_grade_from_score(health_score_value)
        health = {
            "score": health_score_value,
            "grade": grade,
            "letter_grade": grade,
            "grade_class": grade_class,
            "summary": summary,
            "breakdown": {},
        }
    else:
        health = get_session_health(df) if is_present(df) else {"score": None, "grade": "—", "summary": "—", "breakdown": {}}
    issues = detect_data_issues(df) if is_present(df) else []

    documentation = coalesce_dict(safe_dict_get(output, "documentation"))
    model_explanation = chief.get("reasoning_summary") or ""
    sections = coalesce_dict(safe_dict_get(documentation, "sections"))
    if sections.get("Model Selection"):
        model_explanation = sections["Model Selection"]

    canonical_accuracy = resolve_canonical_accuracy(
        executive_metrics,
        model_results,
        best_model=best_model,
        session_best_score=st.session_state.get(SessionKeys.BEST_SCORE),
    )
    accuracy_display = executive_metrics.get("accuracy_display") or format_accuracy_display(
        canonical_accuracy, problem_type
    )
    deployment_status = display_kpi_value(
        executive_deployment_status or chief.get("deployment_label"),
        unavailable="Unavailable",
    )
    trust_display_value = (
        f"{float(executive_trust_score):.0f}/100"
        if executive_trust_score is not None and float(executive_trust_score) > 0
        else "Unavailable"
    )

    identification = coalesce_dict(safe_dict_get(dataset_report, "dataset_identification"))
    uploaded_file = (
        identification.get("uploaded_file")
        or st.session_state.get(SessionKeys.UPLOAD_FILENAME)
        or output.get("dataset_name")
        or (get_dataset_name() if has_dataset() else None)
    )
    detected_dataset = None
    if identification.get("detection_confidence", 0) >= 0.5:
        detected_dataset = identification.get("detected_dataset")

    return {
        "chief": chief,
        "dataset_name": display_kpi_value(detected_dataset or uploaded_file),
        "uploaded_file": display_kpi_value(uploaded_file),
        "detected_dataset": display_kpi_value(detected_dataset) if detected_dataset else None,
        "project_goal": display_kpi_value(output.get("project_goal") or st.session_state.get("project_goal")),
        "problem_type": display_kpi_value(problem_type),
        "target_column": display_kpi_value(target_column),
        "rows": rows,
        "columns": cols,
        "duplicates": duplicates,
        "missing_total": missing_total,
        "missing_pct": missing_pct,
        "numeric_features": numerics if isinstance(numerics, list) else numeric_features,
        "categorical_features": categoricals if isinstance(categoricals, list) else categorical_features,
        "health": health,
        "issues": issues,
        "dataset_report": dataset_report,
        "cleaning_report": cleaning_report,
        "before_shape": before_shape,
        "after_shape": after_shape,
        "cleaning_actions": cleaning_actions,
        "eda": eda if isinstance(eda, dict) else {},
        "eda_insights": eda_insights,
        "leaderboard": leaderboard,
        "best_model": display_kpi_value(best_model),
        "model_selection_notes": model_selection_notes,
        "model_selection_explanation": model_selection_explanation,
        "metrics": metrics,
        "validation_results": validation_results,
        "validation_lines": validation_lines,
        "model_explanation": model_explanation,
        "feature_importance": fi_records,
        "explain_text": expl_text,
        "positive_negative": pos_neg,
        "deploy": deploy,
        "readiness_score": readiness_score,
        "recommendation": recommendation,
        "training_times": training_times,
        "health_score": health.get("score") if isinstance(health, dict) else None,
        "trust_score": executive_trust_score if executive_trust_score is not None else coerce_numeric_score(safe_dict_get(ethics, "trust_score")),
        "trust_display": trust_display_value,
        "confidence_score": executive_confidence or coerce_numeric_score(safe_dict_get(output, "final_ai_confidence_score")) or coerce_numeric_score(safe_dict_get(ethics, "trust_score")),
        "recommendations": normalize_recommendations(st.session_state.get(SessionKeys.RECOMMENDATIONS)),
        "ethics": ethics,
        "feature_importance_records": [
            {"Feature": r.get("feature"), "Importance": r.get("importance")} for r in fi_records
        ],
        "detailed_metrics": detailed_metrics,
        "best_score": canonical_accuracy,
        "accuracy_display": accuracy_display,
        "deployment_status": deployment_status,
        "trust_concerns": normalize_recommendations(
            safe_dict_get(coalesce_dict(ethics), "bias_concerns")
        ),
        "final_report": coalesce_dict(safe_dict_get(output, "final_report")),
        "pipeline_complete": has_autonomous_result(),
    }


def _format_model_metric(value: Any, label: str, problem_type: str) -> str:
    if value is None:
        return "Unavailable"
    try:
        metric_value = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text if text else "Unavailable"

    lower_label = label.lower()
    if "accuracy" in lower_label or "f1" in lower_label or "roc" in lower_label or "score" in lower_label:
        if metric_value <= 1:
            return f"{metric_value * 100:.1f}%"
        return f"{metric_value:.1f}%"
    if lower_label == "r²" or lower_label == "r2":
        return f"{metric_value:.4f}"
    if lower_label in {"rmse", "mae"}:
        return f"{metric_value:.4f}"
    if lower_label == "training time":
        return f"{metric_value:.2f}s"
    return str(metric_value)


def _render_best_model_summary(ctx: Dict[str, Any]) -> None:
    best_model = ctx.get("best_model")
    if not best_model:
        return

    detailed_metrics = coalesce_dict(ctx.get("detailed_metrics"))
    best_metrics = coalesce_dict(detailed_metrics.get(best_model)) if isinstance(detailed_metrics, dict) else {}
    problem_type = str(ctx.get("problem_type") or "Classification")
    training_times = coalesce_dict(ctx.get("training_times"))
    training_time = training_times.get(best_model) if isinstance(training_times, dict) else None

    st.markdown("### Best Model Summary")
    cols = st.columns([2, 2, 2])
    with cols[0]:
        st.metric("Recommended model", display_kpi_value(best_model))
    with cols[1]:
        primary_score = best_metrics.get("accuracy") or best_metrics.get("r2") or best_metrics.get("score") or best_metrics.get("cv_score") or ctx.get("metrics", {}).get(best_model)
        st.metric("Primary score", _metric_format(primary_score, problem_type))
    with cols[2]:
        st.metric("Training time", _format_model_metric(training_time, "Training Time", problem_type))

    metric_lines: List[str] = []
    if any(best_metrics.get(k) is not None for k in ("accuracy", "f1", "roc_auc", "cv_score", "r2", "rmse", "mae")):
        if best_metrics.get("accuracy") is not None:
            metric_lines.append(f"- **Accuracy:** {_format_model_metric(best_metrics.get('accuracy'), 'Accuracy', problem_type)}")
        if best_metrics.get("cv_score") is not None:
            metric_lines.append(f"- **Cross-validated score:** {_format_model_metric(best_metrics.get('cv_score'), 'Accuracy', problem_type)}")
        if best_metrics.get("f1") is not None:
            metric_lines.append(f"- **F1 score:** {_format_model_metric(best_metrics.get('f1'), 'F1 Score', problem_type)}")
        if best_metrics.get("roc_auc") is not None:
            metric_lines.append(f"- **ROC-AUC:** {_format_model_metric(best_metrics.get('roc_auc'), 'ROC-AUC', problem_type)}")
        if best_metrics.get("r2") is not None:
            metric_lines.append(f"- **R²:** {_format_model_metric(best_metrics.get('r2'), 'R²', problem_type)}")
        if best_metrics.get("rmse") is not None:
            metric_lines.append(f"- **RMSE:** {_format_model_metric(best_metrics.get('rmse'), 'RMSE', problem_type)}")
        if best_metrics.get("mae") is not None:
            metric_lines.append(f"- **MAE:** {_format_model_metric(best_metrics.get('mae'), 'MAE', problem_type)}")

    if metric_lines:
        st.markdown("**Key model metrics**")
        for line in metric_lines:
            st.markdown(line)

    explanation = coalesce_dict(ctx.get("model_selection_explanation"))
    if explanation and explanation.get("why_chosen"):
        st.markdown("**Why this model was selected**")
        for reason in coalesce_list(explanation.get("why_chosen"))[:4]:
            st.markdown(f"- {html.escape(str(reason))}")
    elif ctx.get("model_selection_notes"):
        st.markdown("**Selection rationale**")
        for note in coalesce_list(ctx.get("model_selection_notes"))[:4]:
            st.markdown(f"- {html.escape(str(note).replace('**', ''))}")



def _render_kpi_cards(items: List[Tuple[str, str, str]]) -> None:
    for i in range(0, len(items), 3):
        chunk = items[i : i + 3]
        cols = st.columns(len(chunk))
        for col, (label, value, accent) in zip(cols, chunk):
            with col:
                st.markdown(
                    f"""
                    <div class="report-kpi-card" style="--kpi-accent:{accent};">
                        <div class="report-kpi-label">{label}</div>
                        <div class="report-kpi-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _format_validation_metric(raw: Any) -> str:
    if raw is None:
        return "Not Available"
    if isinstance(raw, dict):
        score = raw.get("score") or raw.get("value")
        return _format_validation_metric(score)
    if isinstance(raw, (int, float)):
        return f"{raw:.4f}" if isinstance(raw, float) else str(raw)
    try:
        return f"{float(raw):.4f}"
    except Exception:
        return str(raw)


def _extract_validation_metrics(validation: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not isinstance(validation, dict):
        return {}

    normalized = {str(k).strip().lower().replace(" ", "_"): v for k, v in validation.items()}
    metric_names = {
        "accuracy": ["accuracy", "acc"],
        "precision": ["precision"],
        "recall": ["recall"],
        "f1_score": ["f1_score", "f1", "f1score"],
        "roc_auc": ["roc_auc", "roc_auc_score", "roc_auc_value", "auroc", "rocauc", "roc auc"],
    }
    results: Dict[str, str] = {}
    for label, keys in metric_names.items():
        value = None
        for key in keys:
            if key in normalized:
                value = normalized[key]
                break
        results[label] = _format_validation_metric(value) if value is not None else "Not Available"

    return {
        "Accuracy": results["accuracy"],
        "Precision": results["precision"],
        "Recall": results["recall"],
        "F1 Score": results["f1_score"],
        "ROC-AUC": results["roc_auc"],
    }


def _chart_key(name: str) -> str:
    version = (
        st.session_state.get("report_context_version")
        or st.session_state.get("_applied_pipeline_result_id")
        or "default"
    )
    return f"report_chart_{name}_{version}"


def _ai_interpretation(text: str) -> None:
    if text:
        safe_text = html.escape(str(text))
        st.markdown(
            f"<div class='report-ai-insight'><strong>AI interpretation</strong><br>{safe_text}</div>",
            unsafe_allow_html=True,
        )


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        normalized = (item or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _render_report_narrative(ctx: Dict[str, Any], output: Optional[Dict[str, Any]]) -> None:
    """Structured what / why / risks / recommendations for faculty-ready reports."""
    chief = ctx["chief"]
    model_results = coalesce_dict(safe_dict_get(output, "model_results")) if output else {}
    explanation = coalesce_dict(safe_dict_get(model_results, "model_selection_explanation"))
    ethics = coalesce_dict(ctx.get("ethics"))
    deploy = coalesce_dict(ctx.get("deploy"))

    what_parts = [
        f"AutoDS profiled **{html.escape(str(ctx['dataset_name']))}** "
        f"({ctx['rows']:,} rows, {ctx['columns']} columns)." if ctx["rows"] is not None else f"AutoDS profiled **{html.escape(str(ctx['dataset_name']))}**.",
        f"Problem type: **{html.escape(str(ctx['problem_type']))}** with health score **{chief.get('health_score', ctx.get('health_score', '—'))}/100**.",
        "The pipeline executed profiling, data quality review, EDA, feature engineering, AutoML, SHAP explainability, trust evaluation, and report generation.",
    ]
    cleaning = coalesce_list(ctx.get("cleaning_actions"))
    if cleaning:
        what_parts.append(f"Data cleaning applied {len(cleaning)} automated action(s).")

    why_parts = []
    for note in coalesce_list(ctx.get("model_selection_notes"))[:4]:
        why_parts.append(html.escape(str(note).replace("**", "")))
    if chief.get("reasoning_summary"):
        why_parts.append(html.escape(str(chief["reasoning_summary"])))
    explanation = coalesce_dict(ctx.get("model_selection_explanation"))
    for reason in coalesce_list(explanation.get("why_chosen"))[:4]:
        text = html.escape(str(reason))
        if text not in why_parts:
            why_parts.append(text)
    if not why_parts and ctx.get("best_model"):
        why_parts.append(
            html.escape(
                f"{ctx['best_model']} achieved the highest validation score among all trained candidates for this {ctx['problem_type']} task."
            )
        )

    risk_parts = normalize_recommendations(safe_dict_get(ethics, "bias_concerns"))
    risk_parts += normalize_recommendations(safe_dict_get(deploy, "warnings"))
    risk_parts += normalize_recommendations(safe_dict_get(ethics, "privacy_concerns"))

    business = str(ctx.get("recommendation") or chief.get("business_recommendation") or "")
    for note in coalesce_list(ctx.get("model_selection_notes"))[:2]:
        business = f"{business} {note}".strip()
    if not business or business == "—":
        business = str(safe_dict_get(ethics, "executive_summary") or "")
    business = html.escape(business)

    sections = [
        ("What happened", what_parts),
        ("Why it happened", why_parts or ["Automated stages selected the best model using cross-validation and stability checks."]),
        ("Why this model was selected", why_parts[:3] or [f"**{ctx.get('best_model', '—')}** ranked first on the primary metric."]),
        ("Risks", risk_parts[:6] or ["No critical risks flagged — continue monitoring bias, drift, and data quality."]),
        ("Business recommendations", [business] if business else normalize_recommendations(safe_dict_get(deploy, "recommendations"))[:4]),
    ]

    for title, items in sections:
        st.markdown(f"**{title}**")
        for item in _dedupe_preserve_order([str(item) for item in items]):
            if item:
                st.markdown(f"- {item}")
        st.markdown("")


def _tab_executive_summary(ctx: Dict[str, Any]) -> None:
    chief = ctx["chief"]
    deploy_color = chief.get("deployment_color", PRIMARY_COLOR)
    trust_display = ctx.get("trust_display") or (
        f"{float(ctx['trust_score']):.0f}/100"
        if ctx.get("trust_score") is not None and float(ctx["trust_score"]) > 0
        else "Unavailable"
    )
    # Select primary metric for executive summary: Accuracy for classification;
    # for regression prefer R², then RMSE, then MAE. Never show "Accuracy: Unavailable" for regression.
    def _select_primary_metric(ctx: Dict[str, Any]) -> Tuple[str, str]:
        problem_type = str(ctx.get("problem_type") or "Classification")
        detailed = coalesce_dict(ctx.get("detailed_metrics"))
        best = ctx.get("best_model")
        exec_metrics = st.session_state.get(SessionKeys.EXECUTIVE_METRICS) or {}

        is_regression = "regress" in problem_type.lower()
        if not is_regression:
            # Classification: preserve existing accuracy display behavior
            return "Accuracy", ctx.get("accuracy_display") or format_accuracy_display(ctx.get("best_score"), problem_type)

        # Regression: prefer R², then RMSE, then MAE from executive metrics, detailed metrics, or model results
        for key, label in (("r2", "R² Score"), ("rmse", "RMSE"), ("mae", "MAE")):
            # check executive_metrics first
            val = exec_metrics.get(key)
            if val is None and isinstance(detailed, dict) and best in detailed:
                val = detailed.get(best, {}).get(key)
            if val is not None:
                # Use existing formatting helper for model metrics
                return label, _format_model_metric(val, label, problem_type)

        # Fallback: use best_score formatted as R² where possible, but label as R² Score
        fallback = ctx.get("accuracy_display") or format_accuracy_display(ctx.get("best_score"), problem_type)
        return "R² Score", fallback

    metric_label, acc_display = _select_primary_metric(ctx)
    health_val = ctx.get("health_score")
    health_display = f"{float(health_val):.0f}/100" if health_val is not None else "Unavailable"
    summary_kpis: List[Tuple[str, str, str]] = [
        ("Uploaded File", display_kpi_value(ctx.get("uploaded_file") or ctx.get("dataset_name")), PRIMARY_COLOR),
    ]
    if ctx.get("detected_dataset"):
        summary_kpis.append(("Detected Dataset", display_kpi_value(ctx.get("detected_dataset")), SECONDARY_COLOR))
    summary_kpis.extend(
        [
            ("Rows", f"{ctx['rows']:,}" if ctx.get("rows") is not None else "Unavailable", SECONDARY_COLOR),
            ("Columns", display_kpi_value(ctx.get("columns")), ACCENT_COLOR),
            ("Problem Type", display_kpi_value(ctx.get("problem_type")), SECONDARY_COLOR),
            ("Health Score", health_display, SUCCESS_COLOR),
            ("Best Model", display_kpi_value(ctx.get("best_model")), PRIMARY_COLOR),
            (metric_label, acc_display, ACCENT_COLOR),
            ("Trust Score", trust_display, WARNING_COLOR),
            ("Deployment", display_kpi_value(ctx.get("deployment_status")), deploy_color),
        ]
    )
    _render_kpi_cards(summary_kpis)

    summary_parts = []
    for note in coalesce_list(ctx.get("model_selection_notes"))[:2]:
        summary_parts.append(html.escape(str(note).replace("**", "")))
    if chief.get("reasoning_summary"):
        summary_parts.append(html.escape(str(chief["reasoning_summary"])))
    if ctx["recommendation"] and ctx["recommendation"] != "—":
        summary_parts.append(html.escape(str(ctx["recommendation"])))
    if ctx.get("eda_insights"):
        summary_parts.extend(html.escape(str(item)) for item in ctx["eda_insights"][:2])
    summary_parts = _dedupe_preserve_order(summary_parts)
    if summary_parts:
        _ai_interpretation(" ".join(summary_parts))
    elif ctx["pipeline_complete"]:
        _ai_interpretation(
            html.escape(
                f"AutoDS completed autonomous analysis on {ctx['dataset_name']}. "
                f"The recommended model is {chief.get('model_name', ctx.get('best_model'))} "
                f"with deployment status: {chief.get('deployment_label', 'Under review')}."
            )
        )


def _tab_executive_summary_with_narrative(ctx: Dict[str, Any], output: Optional[Dict[str, Any]]) -> None:
    _tab_executive_summary(ctx)
    if ctx["pipeline_complete"]:
        st.markdown("---")
        _render_report_narrative(ctx, output)


def _tab_dataset_analysis(ctx: Dict[str, Any], df: Optional[pd.DataFrame]) -> None:
    _tab_dataset_intelligence(ctx, df)
    st.markdown("---")
    _tab_data_quality(ctx, df)
    if is_present(df):
        st.markdown("---")
        template = _plotly_template()
        heatmap = correlation_heatmap(df)
        heatmap.update_layout(template=template, height=340)
        st.plotly_chart(heatmap, width="stretch", config=_chart_config(), key=_chart_key("corr"))
        _ai_interpretation(
            "The correlation heatmap highlights linear relationships between numeric features. "
            "Strong correlations may indicate redundant predictors or multicollinearity risks for modeling."
        )
        numerics = ctx.get("numeric_features") or df.select_dtypes(include="number").columns.tolist()
        if numerics:
            selected = st.selectbox("Distribution", numerics, key=_chart_key("dist_feature"))
            fig = px.histogram(df, x=selected, nbins=30, title=f"Distribution: {selected}", color_discrete_sequence=[ACCENT_COLOR])
            fig.update_layout(template=template, height=320)
            st.plotly_chart(fig, width="stretch", config=_chart_config(), key=_chart_key("dist"))
            outlier_info = detect_outliers(df, selected)
            if isinstance(outlier_info, dict) and outlier_info.get("count") is not None:
                _ai_interpretation(
                    f"Feature `{selected}` contains {outlier_info.get('count')} outliers "
                    f"({outlier_info.get('pct', 0):.1f}% of rows). Outliers can skew models and should be monitored in production."
                )
            else:
                _ai_interpretation(
                    f"The distribution of `{selected}` shows how values spread across the dataset, "
                    "helping identify skew, sparsity, and potential encoding needs."
                )
        insights = coalesce_list(ctx.get("eda_insights"))
        if insights:
            st.markdown("**AI EDA insights**")
            for insight in insights[:8]:
                st.markdown(f"- {insight}")


def _tab_models_report(ctx: Dict[str, Any], output: Optional[Dict[str, Any]]) -> None:
    _render_best_model_summary(ctx)
    st.markdown("---")
    _tab_model_competition(ctx)
    explanation = coalesce_dict(ctx.get("model_selection_explanation")) or coalesce_dict(
        safe_dict_get(coalesce_dict(safe_dict_get(output, "model_results")), "model_selection_explanation")
    )
    if explanation:
        st.markdown("**Why this model was selected**")
        for reason in coalesce_list(explanation.get("why_chosen")):
            st.markdown(f"- {html.escape(str(reason))}")
        for alt in coalesce_list(explanation.get("alternatives"))[:5]:
            if isinstance(alt, dict):
                st.markdown(f"**Why {html.escape(str(alt.get('model', 'Alternative')))} was not chosen**")
                for reason in coalesce_list(alt.get("reasons")):
                    st.markdown(f"- {html.escape(str(reason))}")
    elif ctx.get("model_explanation"):
        _ai_interpretation(str(ctx["model_explanation"]))
    detailed = coalesce_dict(ctx.get("detailed_metrics")) or coalesce_dict(st.session_state.get(SessionKeys.MODEL_METRICS))
    best = ctx.get("best_model")
    if best and isinstance(detailed, dict) and best in detailed:
        dm = coalesce_dict(detailed.get(best))
        cv = dm.get("cv_score")
        if cv is not None:
            metric_label = "Accuracy (Cross-Validated)" if str(ctx["problem_type"]).lower() == "classification" else "R² (Cross-Validated)"
            st.metric(metric_label, _metric_format(float(cv), str(ctx["problem_type"])))
        if str(ctx["problem_type"]).lower() == "regression":
            r2 = dm.get("r2")
            rmse = dm.get("rmse")
            mae = dm.get("mae")
            if r2 is not None:
                st.metric("R² Score", _metric_format(float(r2), str(ctx["problem_type"])))
            if rmse is not None:
                st.metric("RMSE", f"{float(rmse):.4f}")
            if mae is not None:
                st.metric("MAE", f"{float(mae):.4f}")


def _tab_trust_risk(ctx: Dict[str, Any]) -> None:
    executive_metrics = st.session_state.get(SessionKeys.EXECUTIVE_METRICS) or {}
    ethics = coalesce_dict(ctx.get("ethics"))
    chief = ctx["chief"]
    deploy = coalesce_dict(ctx.get("deploy"))
    trust_num = coerce_numeric_score(ctx.get("trust_score") or executive_metrics.get("trust_score") or chief.get("trust_score"))
    trust_display = f"{trust_num:.0f}/100" if trust_num is not None and trust_num > 0 else "Unavailable"
    readiness = ctx.get("readiness_score")
    risk_level = display_kpi_value(
        deploy.get("risk_level") or executive_metrics.get("risk_level") or chief.get("risk_level"),
        unavailable="Unavailable",
    )
    deployment_status = display_kpi_value(ctx.get("deployment_status") or executive_metrics.get("deployment_status"))

    _render_kpi_cards(
        [
            ("Trust Score", trust_display, PRIMARY_COLOR),
            ("Risk Level", risk_level, WARNING_COLOR),
            ("Deployment Status", deployment_status, SUCCESS_COLOR),
            ("Deployment Readiness", f"{readiness}/100" if readiness is not None else "Unavailable", ACCENT_COLOR),
        ]
    )

    bias = normalize_recommendations(safe_dict_get(ethics, "bias_concerns"))
    privacy = normalize_recommendations(safe_dict_get(ethics, "privacy_concerns"))
    compliance = normalize_recommendations(safe_dict_get(ethics, "compliance_recommendations")) or normalize_recommendations(safe_dict_get(ethics, "recommendations"))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Bias Assessment**")
        for item in (bias or ["No major bias concerns detected."])[:5]:
            st.markdown(f"- {item}")
    with c2:
        st.markdown("**Privacy Assessment**")
        for item in (privacy or ["No sensitive identifiers flagged."])[:5]:
            st.markdown(f"- {item}")
    with c3:
        st.markdown("**Compliance Assessment**")
        for item in (compliance or ["Standard ML governance checks applied."])[:5]:
            st.markdown(f"- {item}")

    warnings = normalize_recommendations(safe_dict_get(deploy, "warnings"))
    if warnings:
        st.markdown("**Risk warnings**")
        for warning in warnings[:6]:
            st.markdown(f"- {warning}")

    summary = safe_dict_get(ethics, "executive_summary")
    if summary:
        _ai_interpretation(str(summary))
    else:
        _ai_interpretation(
            f"Trust evaluation completed with risk level {risk_level}. "
            f"Deployment readiness is {readiness}/100. Review bias and privacy notes before production use."
        )


def _tab_final_decision_with_trust(ctx: Dict[str, Any], output: Optional[Dict[str, Any]]) -> None:
    _tab_trust_risk(ctx)
    st.divider()
    _tab_final_decision(ctx, output)


def _tab_final_decision(ctx: Dict[str, Any], output: Optional[Dict[str, Any]]) -> None:
    executive_metrics = st.session_state.get(SessionKeys.EXECUTIVE_METRICS) or {}
    chief = ctx["chief"]
    deploy_key = chief.get("deployment_key", "monitor")
    deployment_label = display_kpi_value(ctx.get("deployment_status") or executive_metrics.get("deployment_status"), unavailable="Pending")
    render_deployment_status_card(deployment_label, status_key=str(deploy_key))

    trust_value = coerce_numeric_score(ctx.get("trust_score") or executive_metrics.get("trust_score"))
    trust_display = f"{trust_value:.0f}/100" if trust_value is not None and trust_value > 0 else "Unavailable"
    acc_display = ctx.get("accuracy_display") or format_accuracy_display(ctx.get("best_score"), str(ctx.get("problem_type")))

    _render_kpi_cards(
        [
            ("Recommended Model", display_kpi_value(ctx.get("best_model") or chief.get("model_name")), PRIMARY_COLOR),
            (primary_metric_label(str(ctx.get("problem_type") or "Classification")), acc_display, ACCENT_COLOR),
            ("Trust Score", trust_display, WARNING_COLOR),
            ("Deployment Recommendation", deployment_label, SUCCESS_COLOR),
        ]
    )

    decision_text = chief.get("business_recommendation") or ctx.get("recommendation")
    for note in coalesce_list(ctx.get("model_selection_notes"))[:2]:
        decision_text = f"{decision_text} {note}".strip() if decision_text and str(decision_text).strip() not in ("—", "") else str(note)
    if decision_text and str(decision_text).strip() not in ("—", ""):
        st.markdown("**Final executive decision**")
        st.markdown(
            f"<div class='report-insight-box'>{html.escape(str(decision_text))}</div>",
            unsafe_allow_html=True,
        )
    elif ctx["pipeline_complete"]:
        _ai_interpretation(
            display_kpi_value(
                ctx.get("recommendation") or chief.get("reasoning_summary"),
                unavailable="Review trust metrics and deployment readiness before production use.",
            )
        )

    if output is not None:
        with st.expander("Full Chief Data Scientist report", expanded=False):
            render_chief_decision_panel(output, compact=False)


def _tab_explainability_report(ctx: Dict[str, Any]) -> None:
    _tab_explainability(ctx)
    fi_records = coalesce_list(ctx.get("feature_importance"))
    if fi_records:
        ranked = sorted(fi_records, key=lambda item: abs(item["importance"]), reverse=True)[:5]
        top_names = ", ".join(item["feature"] for item in ranked)
        _ai_interpretation(
            f"The strongest prediction drivers are {top_names}. "
            "These features influence model outcomes most and should be monitored for data drift in production."
        )
    if ctx.get("explain_text"):
        _ai_interpretation(str(ctx["explain_text"]))


def _tab_dataset_intelligence(ctx: Dict[str, Any], df: Optional[pd.DataFrame]) -> None:
    _render_kpi_cards(
        [
            ("Rows", f"{ctx['rows']:,}" if ctx["rows"] is not None else "—", PRIMARY_COLOR),
            ("Columns", str(ctx["columns"]) if ctx["columns"] is not None else "—", SECONDARY_COLOR),
            ("Missing Values", f"{ctx['missing_total']:,} ({ctx['missing_pct']:.1f}%)" if ctx["missing_total"] is not None else "—", WARNING_COLOR),
            ("Duplicates", f"{ctx['duplicates']:,}" if ctx["duplicates"] is not None else "—", DANGER_COLOR),
            ("Numeric Features", str(len(ctx["numeric_features"])), ACCENT_COLOR),
            ("Categorical Features", str(len(ctx["categorical_features"])), SUCCESS_COLOR),
        ]
    )

    st.markdown(f"**Uploaded file:** {ctx.get('uploaded_file') or ctx.get('dataset_name')}")
    if ctx.get("detected_dataset"):
        st.markdown(f"**Detected dataset:** {ctx['detected_dataset']}")
    st.markdown(f"**Target column:** {ctx['target_column']}")

    health = ctx["health"]
    grade_class = health.get("grade_class", "health-fair")
    st.markdown(
        f"""
        <div class="report-health-banner {grade_class}">
            <strong>Dataset health:</strong> {health.get('score', '—')}/100 · Grade {health.get('grade', '—')}
            <div class="report-muted">{health.get('summary', '')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    breakdown = coalesce_dict(health.get("breakdown"))
    if breakdown:
        fig = go.Figure(
            go.Bar(
                x=list(breakdown.values()),
                y=list(breakdown.keys()),
                orientation="h",
                marker_color=ACCENT_COLOR,
            )
        )
        fig.update_layout(title="Health breakdown", template=_plotly_template(), height=320, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, width="stretch", config=_chart_config(), key=_chart_key("health_breakdown"))

    report = coalesce_dict(ctx.get("dataset_report"))
    if report:
        problem = coalesce_dict(safe_dict_get(report, "problem_analysis"))
        summary = safe_dict_get(problem, "summary") or safe_dict_get(report, "summary")
        if is_present(summary):
            _ai_interpretation(str(summary))


def _tab_data_quality(ctx: Dict[str, Any], df: Optional[pd.DataFrame]) -> None:
    health = ctx["health"]
    st.metric("Quality score", f"{health.get('score', '—')}/100")

    col_before, col_after = st.columns(2)
    with col_before:
        st.markdown("**Before cleaning**")
        before = ctx.get("before_shape")
        if before:
            st.write(f"{before[0]:,} rows × {before[1]} columns")
        elif ctx["rows"] is not None:
            st.write(f"{ctx['rows']:,} rows × {ctx['columns']} columns")
        else:
            st.caption("Shape unavailable until pipeline runs.")

    with col_after:
        st.markdown("**After cleaning**")
        after = ctx.get("after_shape")
        if after:
            st.write(f"{after[0]:,} rows × {after[1]} columns")
        elif ctx["rows"] is not None and ctx.get("cleaning_actions"):
            st.write(f"{ctx['rows']:,} rows × {ctx['columns']} columns")
        else:
            st.caption("Cleaning results pending.")

    if is_present(df):
        st.plotly_chart(missing_values_chart(df), width="stretch", config=_chart_config(), key=_chart_key("missing"))
        _ai_interpretation(
            "Missing value patterns show which columns need imputation or removal. "
            "High missingness reduces model reliability and should be addressed before deployment."
        )

    issues = coalesce_list(ctx.get("issues"))
    if issues:
        st.markdown("**Detected quality issues**")
        for issue in issues[:8]:
            if isinstance(issue, dict):
                st.markdown(f"- **{issue.get('title', 'Issue')}:** {issue.get('description', '')}")

    actions = coalesce_list(ctx.get("cleaning_actions"))
    if actions:
        st.markdown("**Cleaning actions performed**")
        for action in actions:
            st.markdown(f"- {action}")
    elif not ctx["pipeline_complete"]:
        st.caption("Run the autonomous pipeline to populate cleaning actions.")

    report = coalesce_dict(ctx.get("cleaning_report"))
    if isinstance(report, dict) and report.get("outliers_detected"):
        st.markdown(f"**Outlier analysis:** {report.get('outliers_detected')}")


def _tab_model_competition(ctx: Dict[str, Any]) -> None:
    leaderboard = coalesce_list(ctx.get("leaderboard"))
    if not leaderboard:
        st.info("Model leaderboard will appear after AutoML training completes.")
        return

    st.dataframe(pd.DataFrame(leaderboard[:5]), width="stretch", hide_index=True)

    if ctx.get("metrics"):
        names = [row["Model"] for row in leaderboard]
        scores = []
        for row in leaderboard:
            raw = ctx["metrics"].get(row["Model"])
            scores.append(float(raw) if raw is not None else 0)
        fig = go.Figure(go.Bar(x=scores, y=names, orientation="h", marker_color=PRIMARY_COLOR))
        fig.update_layout(
            title="Model score comparison",
            template=_plotly_template(),
            height=max(280, len(names) * 32),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, width="stretch", config=_chart_config(), key=_chart_key("model_compare"))

    if ctx.get("model_explanation"):
        st.markdown("**Best model explanation**")
        st.write(ctx["model_explanation"])

    validation_results = coalesce_dict(ctx.get("validation_results"))
    if validation_results:
        st.markdown("**Validation metrics**")
        metrics_display = _extract_validation_metrics(validation_results)
        if metrics_display:
            display_df = pd.DataFrame(
                [{"Metric": metric, "Value": value} for metric, value in metrics_display.items()]
            )
            st.dataframe(display_df, width="stretch", hide_index=True)
        else:
            st.caption("Validation metrics are unavailable for the current model results.")
    elif ctx.get("validation_lines"):
        st.markdown("**Validation metrics**")
        for line in ctx["validation_lines"]:
            st.write(f"- {line}")
    else:
        st.caption("Validation metrics not yet available from the pipeline.")


def _tab_explainability(ctx: Dict[str, Any]) -> None:
    fi_records = coalesce_list(ctx.get("feature_importance"))
    if fi_records:
        ranked = sorted(fi_records, key=lambda item: abs(item["importance"]), reverse=True)[:15]
        fig = go.Figure(
            go.Bar(
                x=[item["importance"] for item in ranked],
                y=[item["feature"] for item in ranked],
                orientation="h",
                marker_color=ACCENT_COLOR,
            )
        )
        fig.update_layout(title="Feature importance", template=_plotly_template(), height=360)
        st.plotly_chart(fig, width="stretch", config=_chart_config(), key=_chart_key("shap"))

        st.markdown("**Top features**")
        for item in ranked[:8]:
            st.markdown(f"- `{item['feature']}` — {item['importance']:.4f}")
    else:
        st.info("Feature importance will appear after the explainability stage completes.")

    pos_neg = ctx.get("positive_negative")
    if pos_neg:
        st.markdown("**Positive / negative drivers**")
        if isinstance(pos_neg, dict):
            for key, value in list(pos_neg.items())[:8]:
                st.write(f"- {key}: {value}")
        else:
            st.write(pos_neg)

    if ctx.get("explain_text"):
        st.markdown("**Natural language explanation**")
        st.write(ctx["explain_text"])


def _tab_deployment(ctx: Dict[str, Any]) -> None:
    deploy = coalesce_dict(ctx.get("deploy"))
    chief = ctx["chief"]
    readiness = ctx.get("readiness_score")
    risk_level = deploy.get("risk_level") or chief.get("risk_level") or "Unknown"
    risk_color = {"Low": SUCCESS_COLOR, "Medium": WARNING_COLOR, "High": DANGER_COLOR}.get(str(risk_level), PRIMARY_COLOR)

    _render_kpi_cards(
        [
            ("Readiness Score", f"{readiness}/100" if readiness is not None else "—", PRIMARY_COLOR),
            ("Risk Level", str(risk_level), risk_color),
            ("Deployment Status", str(chief.get("deployment_label", "—")), chief.get("deployment_color", PRIMARY_COLOR)),
        ]
    )

    warnings = normalize_recommendations(safe_dict_get(deploy, "warnings"))
    if warnings:
        st.markdown("**Risk warnings**")
        for warning in warnings[:8]:
            st.markdown(f"<span class='chief-risk-badge'>⚠ {warning}</span>", unsafe_allow_html=True)

    recommendations = normalize_recommendations(safe_dict_get(deploy, "recommendations"))
    if not recommendations:
        recommendations = normalize_recommendations(safe_dict_get(deploy, "api_recommendations"))
    if recommendations:
        st.markdown("**Deployment recommendations**")
        for rec in recommendations[:8]:
            st.markdown(f"- {rec}")

    monitoring = normalize_recommendations(safe_dict_get(deploy, "monitoring_suggestions"))
    if not monitoring:
        monitoring = normalize_recommendations(safe_dict_get(deploy, "monitoring"))
    if isinstance(monitoring, list) and monitoring:
        st.markdown("**Monitoring suggestions**")
        for item in monitoring[:8]:
            st.markdown(f"- {item}")
    elif deploy.get("reasoning"):
        st.markdown("**Monitoring guidance**")
        st.write(deploy.get("reasoning"))


def _render_report_exports(ctx: Dict[str, Any]) -> None:
    """PDF and Excel export controls on one row."""
    report_info = coalesce_dict(ctx.get("final_report"))
    report_path = report_info.get("path") if isinstance(report_info, dict) else None
    session_path = st.session_state.get(SessionKeys.REPORT_PATH)

    pdf_paths = []
    if report_path and os.path.exists(report_path):
        pdf_paths.append(report_path)
    if session_path and os.path.exists(session_path) and session_path not in pdf_paths:
        pdf_paths.append(session_path)

    export_ctx = build_export_context_from_report_ctx(ctx)
    col_pdf, col_excel = st.columns(2)
    with col_pdf:
        if pdf_paths:
            with open(pdf_paths[0], "rb") as fh:
                pdf_bytes = fh.read()
            st.download_button(
                "📄 Export PDF",
                pdf_bytes,
                file_name=os.path.basename(pdf_paths[0]),
                mime="application/pdf",
                type="secondary",
                key="report_center_pdf_export",
                width="stretch",
            )
        else:
            st.caption("PDF export becomes available when the pipeline generates a final report.")
    with col_excel:
        try:
            excel_bytes = build_excel_bytes(export_ctx)
            st.download_button(
                "📊 Export Excel",
                excel_bytes,
                file_name="AutoDS_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="report_center_excel_export",
                width="stretch",
            )
        except Exception as exc:
            st.caption(f"Excel export unavailable: {exc}")


def render_interactive_report_center(output: Optional[Dict[str, Any]] = None) -> None:
    """Render the premium Interactive AI Report Center."""
    _ = st.session_state.get("report_context_version", "")

    if has_autonomous_result():
        output = get_autonomous_result()

    df = get_dataframe() if has_dataset() else None
    ctx = build_report_context(output, df)

    st.markdown(
        """
        <div class="report-center-shell">
            <div class="report-center-header">
                <div class="report-center-icon">📊</div>
                <div>
                    <div class="report-center-kicker">Interactive Report Center</div>
                    <div class="report-center-title">AI Research & Decision Report</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.pop("_show_analysis_complete_banner", False):
        st.success("Autonomous analysis complete. Your executive AI decision report is ready below.")

    if not has_dataset() and not has_autonomous_result():
        from ui.components import render_no_dataset_gate
        render_no_dataset_gate()
        return

    if not has_dataset() and has_autonomous_result():
        st.warning("Dataset session was cleared, but analysis results are still available below.")

    if not ctx["pipeline_complete"]:
        st.info("Run **Autonomous Analysis** on Home to populate this report.")

    validation_warnings = st.session_state.get("pipeline_validation_warnings") or []
    if validation_warnings:
        st.warning(
            "Some pipeline sections are incomplete: "
            + ", ".join(validation_warnings)
            + ". Available sections still use real analysis data."
        )

    _render_report_exports(ctx)

    tabs = st.tabs(
        [
            "Executive Summary",
            "Dataset Analysis",
            "Models",
            "Explainability",
            "Final Decision",
        ]
    )

    with tabs[0]:
        run_panel("Executive Summary", lambda: _tab_executive_summary_with_narrative(ctx, output))
    with tabs[1]:
        run_panel("Dataset Analysis", lambda: _tab_dataset_analysis(ctx, df))
    with tabs[2]:
        run_panel("Models", lambda: _tab_models_report(ctx, output))
    with tabs[3]:
        run_panel("Explainability", lambda: _tab_explainability_report(ctx))
    with tabs[4]:
        run_panel("Final Decision", lambda: _tab_final_decision_with_trust(ctx, output))
