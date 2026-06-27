"""Premium AI Chief Data Scientist Decision panel for the Command Center."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import ACCENT_COLOR, DANGER_COLOR, PRIMARY_COLOR, SUCCESS_COLOR, WARNING_COLOR
from utils.health_score import compute_health_score, detect_data_issues
from utils.safe_checks import (
    coerce_numeric_score,
    coalesce_dict,
    coalesce_list,
    feature_importance_as_dict,
    is_present,
    normalize_feature_importance,
    normalize_recommendations,
    safe_dict_get,
    resolve_canonical_accuracy,
    format_accuracy_display,
)
from utils.session_manager import SessionKeys, get_problem_type, get_dataframe, has_autonomous_result


def _format_metric(score: Optional[float], problem_type: str) -> Tuple[str, str]:
    if score is None:
        return "Score", "—"
    label = "R²" if str(problem_type).lower() == "regression" else "Accuracy"
    if score <= 1:
        return label, f"{score * 100:.1f}%"
    return label, f"{score:.4f}"


def _format_optional_metric(value: Optional[float]) -> str:
    if value is None:
        return "—"
    if value <= 1:
        return f"{value * 100:.1f}%"
    return f"{value:.4f}"


def _extract_f1_score(output: Optional[Dict[str, Any]]) -> Optional[float]:
    if not output:
        return None
    for source in (
        coalesce_list(safe_dict_get(output, "model_comparison")),
        coalesce_list(safe_dict_get(output, "validation_results")),
    ):
        if isinstance(source, list):
            for entry in reversed(source):
                if not isinstance(entry, dict):
                    continue
                metrics = entry.get("metrics") if "metrics" in entry else entry
                if isinstance(metrics, dict):
                    for key in ("f1", "f1_score", "F1"):
                        if key in metrics and metrics[key] is not None:
                            return float(metrics[key])
        elif isinstance(source, dict):
            for key in ("f1", "f1_score", "F1"):
                if key in source and source[key] is not None:
                    return float(source[key])
    registry = coalesce_dict(safe_dict_get(output, "model_registry_entry"))
    if not registry:
        registry = coalesce_dict(safe_dict_get(output, "model_registry"))
    reg_metrics = registry.get("metrics") if isinstance(registry, dict) else {}
    if isinstance(reg_metrics, dict):
        for key in ("f1", "f1_score", "F1"):
            if key in reg_metrics and reg_metrics[key] is not None:
                return float(reg_metrics[key])
    return None


def _deployment_status(final_score: Optional[float], risk_level: str, recommendation: str) -> Tuple[str, str, str]:
    risk = (risk_level or "Unknown").lower()
    score = final_score if final_score is not None else 0
    if score >= 80 and risk == "low":
        return "ready", "🟢 Production Ready", SUCCESS_COLOR
    if score >= 60 or risk == "medium":
        return "monitor", "🟡 Needs Monitoring", WARNING_COLOR
    return "not_ready", "🔴 Not Ready", DANGER_COLOR


def build_chief_decision_data(output: Optional[Dict[str, Any]], df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Assemble decision panel data from pipeline output and session state."""
    from utils.model_ranking import build_best_model_consistency_notes, rank_models_by_composite

    model_results = coalesce_dict(safe_dict_get(output, "model_results")) if output else {}
    metrics = coalesce_dict(safe_dict_get(model_results, "metrics")) or st.session_state.get(SessionKeys.RESULTS) or {}
    detailed_metrics = coalesce_dict(safe_dict_get(model_results, "detailed_metrics")) or coalesce_dict(st.session_state.get(SessionKeys.MODEL_METRICS))
    problem_type = get_problem_type(output)
    best_from_results = model_results.get("best_model") or st.session_state.get(SessionKeys.BEST_MODEL_NAME)
    explanation = coalesce_dict(safe_dict_get(model_results, "model_selection_explanation"))
    selection_notes = build_best_model_consistency_notes(
        best_from_results,
        detailed_metrics,
        str(problem_type),
        metrics,
        explanation,
    )

    leaderboard: List[Dict[str, str]] = []
    ranked = rank_models_by_composite(detailed_metrics, str(problem_type), metrics)
    if ranked:
        for rank, (name, score) in enumerate(ranked, start=1):
            pct = f"{score * 100:.1f}%" if score <= 1 else f"{score:.4f}"
            leaderboard.append({"rank": str(rank), "model": name, "score": pct})
    elif isinstance(metrics, dict) and metrics:
        for rank, (name, score) in enumerate(sorted(metrics.items(), key=lambda x: x[1], reverse=True), start=1):
            pct = f"{score * 100:.1f}%" if score <= 1 else f"{score:.4f}"
            leaderboard.append({"rank": str(rank), "model": name, "score": pct})

    # === READ FROM EXECUTIVE_METRICS FIRST (Single Source of Truth) ===
    executive_metrics = st.session_state.get(SessionKeys.EXECUTIVE_METRICS) or {}
    if executive_metrics:
        best_model = executive_metrics.get("best_model") or best_from_results
        notes = build_best_model_consistency_notes(
            best_model,
            detailed_metrics,
            str(problem_type),
            metrics,
            explanation,
        )
        business_rec = executive_metrics.get("final_decision", {}).get("recommendation", "Review metrics and proceed with caution.")
        if notes:
            business_rec = f"{business_rec} {' '.join(notes[:2])}".strip()
        reasoning = f"Pipeline recommendation: {executive_metrics.get('deployment_status', 'Pending')}"
        if notes:
            reasoning = f"{reasoning}. {notes[0]}"
        # Use authoritative executive metrics for consistency across all pages
        problem_type = get_problem_type(output)
        primary_label = "R² Score" if str(problem_type).lower().startswith("regress") else "Accuracy"
        # Prefer explicit display string if present
        primary_value = executive_metrics.get("accuracy_display")
        # For regression, prefer explicit r2 fields if available
        if not primary_value and str(problem_type).lower().startswith("regress"):
            r2 = executive_metrics.get("r2") or executive_metrics.get("r2_score")
            if r2 is None:
                mr = coalesce_dict(safe_dict_get(executive_metrics, "model_results"))
                mr_metrics = coalesce_dict(safe_dict_get(mr, "metrics"))
                r2 = mr_metrics.get("r2") if isinstance(mr_metrics, dict) else None
            if r2 is not None:
                primary_value = format_accuracy_display(r2, problem_type)
        # Fallback to canonical accuracy/r2 resolution
        if not primary_value:
            primary_value = format_accuracy_display(
                resolve_canonical_accuracy(executive_metrics, coalesce_dict(safe_dict_get(executive_metrics, "model_results")), best_model=best_model),
                problem_type,
            )
        return {
            "pipeline_complete": has_autonomous_result(),
            "model_name": best_model or "—",
            "model_version": executive_metrics.get("model_version", "—"),
            "metric_label": primary_label,
            "metric_value": primary_value or "—",
            "f1_display": "—",
            "problem_type": problem_type or "Classification",
            "confidence": executive_metrics.get("confidence_score"),
            "final_score": executive_metrics.get("confidence_score"),
            "health_score": executive_metrics.get("health_score"),
            "health_grade": "Excellent" if executive_metrics.get("health_score", 0) >= 90 
                           else "Good" if executive_metrics.get("health_score", 0) >= 75 
                           else "Fair" if executive_metrics.get("health_score", 0) >= 60 
                           else "Poor",
            "health_summary": f"Dataset health: {executive_metrics.get('health_score', 0):.1f}/100",
            "intelligence_score": executive_metrics.get("health_score"),
            "missing_total": 0,
            "missing_pct": 0.0,
            "outlier_note": None,
            "issues": [],
            "leaderboard": leaderboard,
            "validation_lines": [],
            "overfitting_lines": ["Model validation details pending."],
            "stability_lines": ["Training stability assessment pending."],
            "explain_lines": ["Explainability results will appear after SHAP stage completes."],
            "strengths": [f"✓ {best_model or 'Model'} selected as top performer",
                         f"✓ Trust score: {executive_metrics.get('trust_score', 0):.1f}/100"],
            "risks": ["⚠ Continue monitoring for data drift in production."],
            "business_recommendation": business_rec,
            "reasoning_summary": reasoning,
            "model_selection_notes": notes,
            "deployment_key": "ready" if executive_metrics.get("deployment_status") == "Production Ready" else "monitor" if "Monitoring" in executive_metrics.get("deployment_status", "") else "not_ready",
            "deployment_label": executive_metrics.get("deployment_status", "Pending"),
            "deployment_color": "#10B981" if executive_metrics.get("deployment_status") == "Production Ready" else "#F59E0B" if "Monitoring" in executive_metrics.get("deployment_status", "") else "#EF4444",
            "risk_level": executive_metrics.get("risk_level", "Unknown"),
            "trust_score": executive_metrics.get("trust_score"),
        }
    
    # === FALLBACK: Original logic if no executive_metrics ===
    best_name = best_from_results

    registry = (output or {}).get("model_registry_entry") or (output or {}).get("model_registry") or {}
    version = (
        (output or {}).get("best_model_version")
        or registry.get("version")
        or registry.get("model_id")
    )

    best_score = metrics.get(best_name) if best_name and isinstance(metrics, dict) else None
    metric_label, metric_value = _format_metric(best_score, problem_type)
    f1_score = _extract_f1_score(output) if str(problem_type).lower() != "regression" else None
    f1_display = _format_optional_metric(f1_score)

    confidence = (
        (output or {}).get("final_ai_confidence_score")
        or st.session_state.get(SessionKeys.CONFIDENCE_SCORE)
    )

    health = compute_health_score(df) if df is not None and not df.empty else {"score": 0, "grade": "N/A", "summary": "No dataset loaded."}
    issues = detect_data_issues(df) if df is not None and not df.empty else []

    missing_total = int(df.isnull().sum().sum()) if df is not None and not df.empty else 0
    missing_pct = float(df.isnull().mean().mean() * 100) if df is not None and not df.empty else 0.0

    dataset_report = (output or {}).get("dataset_report") or (output or {}).get("dataset_analysis") or {}
    intelligence_score = None
    if isinstance(dataset_report, dict):
        intelligence_score = (dataset_report.get("intelligence_score") or {}).get("score")

    cleaning = (output or {}).get("cleaning_results") or {}
    cleaning_report = cleaning.get("report") if isinstance(cleaning, dict) else {}
    outlier_note = None
    if isinstance(cleaning_report, dict):
        outlier_note = cleaning_report.get("outliers_detected") or cleaning_report.get("outliers")

    explainability = coalesce_dict(safe_dict_get(output, "explainability_results"))
    if not explainability:
        explainability = coalesce_dict(safe_dict_get(output, "xai_results"))
    fi_raw = safe_dict_get(explainability, "feature_importance")
    fi_records = normalize_feature_importance(fi_raw)
    if not fi_records:
        fi_records = normalize_feature_importance(st.session_state.get(SessionKeys.SHAP_IMPORTANCE))
    fi = feature_importance_as_dict(fi_records)
    top_features: List[str] = [item["feature"] for item in fi_records[:5]]

    validation = (output or {}).get("validation_results") or {}
    validation_lines: List[str] = []
    if isinstance(validation, dict) and validation:
        for key, value in validation.items():
            validation_lines.append(f"{key}: {value}")
    elif isinstance(validation, list) and validation:
        for entry in validation[:5]:
            validation_lines.append(str(entry))

    overfitting_lines: List[str] = []
    if isinstance(metrics, dict) and len(metrics) >= 2:
        spread = max(metrics.values()) - min(metrics.values())
        overfitting_lines.append(
            f"Leaderboard spread across {len(metrics)} models: {spread:.4f} "
            f"({'tight cluster — review train vs validation' if spread < 0.03 else 'meaningful separation between candidates'})."
        )
    if isinstance(validation, dict):
        train_score = validation.get("train_score") or validation.get("training_score")
        val_score = validation.get("validation_score") or validation.get("val_score") or validation.get("test_score")
        if train_score is not None and val_score is not None:
            gap = float(train_score) - float(val_score)
            overfitting_lines.append(
                f"Train vs validation gap: {gap:.4f} "
                f"({'possible overfitting' if gap > 0.08 else 'stable generalization signal'})."
            )
    if not overfitting_lines:
        overfitting_lines.append("Cross-validation details unavailable — run the full pipeline for deeper overfitting checks.")

    stability_lines: List[str] = []
    improvement = (output or {}).get("improvement_history") or []
    if isinstance(improvement, list) and improvement:
        stability_lines.extend(str(item) for item in improvement[:4])
    optimization = (output or {}).get("optimization_report") or {}
    if isinstance(optimization, dict) and optimization.get("iterations"):
        stability_lines.append(f"Hyperparameter optimization iterations: {optimization.get('iterations')}")
    if not stability_lines:
        stability_lines.append("Training stability history will appear after model improvement stages complete.")

    explain_lines: List[str] = []
    if top_features:
        explain_lines.append(f"Top drivers: {', '.join(top_features)}.")
    expl_text = explainability.get("explanation") or explainability.get("summary") if isinstance(explainability, dict) else None
    if expl_text and not (top_features and "shap skipped" in str(expl_text).lower()):
        explain_lines.append(str(expl_text))
    if not explain_lines:
        explain_lines.append("Explainability results pending — complete the SHAP stage in the pipeline.")

    strengths: List[str] = []
    generalization_ok = False
    if isinstance(metrics, dict) and len(metrics) >= 2:
        spread = max(metrics.values()) - min(metrics.values())
        if spread >= 0.03:
            generalization_ok = True
    if isinstance(validation, dict):
        train_score = validation.get("train_score") or validation.get("training_score")
        val_score = validation.get("validation_score") or validation.get("val_score") or validation.get("test_score")
        if train_score is not None and val_score is not None and float(train_score) - float(val_score) <= 0.08:
            generalization_ok = True
    if best_score is not None and ((best_score <= 1 and best_score >= 0.70) or best_score >= 0.55):
        generalization_ok = True
    if generalization_ok:
        strengths.append("✓ Good generalization")

    if validation_lines or (isinstance(validation, dict) and validation):
        strengths.append("✓ Stable validation performance")

    if top_features:
        strengths.append("✓ Important features discovered")

    if not strengths and not has_autonomous_result():
        strengths.append("Run Autonomous Analysis to discover model strengths.")
    elif not strengths and best_name:
        strengths.append(f"✓ {best_name} selected as top performer")

    risks: List[str] = []
    deploy = coalesce_dict(safe_dict_get(output, "deployment_readiness"))
    for warning in normalize_recommendations(safe_dict_get(deploy, "warnings")):
        risks.append(f"⚠ {warning}")
    ethics = coalesce_dict(safe_dict_get(output, "ethics_report"))
    if not ethics:
        ethics = coalesce_dict(safe_dict_get(output, "ai_trust_results"))
    trust_score = safe_dict_get(ethics, "trust_score") if ethics else None
    if isinstance(ethics, dict):
        bias = coalesce_dict(safe_dict_get(ethics, "bias_analysis"))
        for concern in normalize_recommendations(safe_dict_get(bias, "bias_concerns")) + normalize_recommendations(safe_dict_get(ethics, "concerns")):
            risks.append(f"⚠ Ethics: {concern}")
    for issue in issues[:6]:
        if isinstance(issue, dict) and issue.get("title") != "No Critical Issues":
            risks.append(f"⚠ {issue.get('title')}: {issue.get('description', '')}")
    stage_errors = (output or {}).get("stage_errors") or []
    for err in stage_errors[:3]:
        if isinstance(err, dict):
            risks.append(f"⚠ Pipeline stage '{err.get('stage', 'unknown')}': {err.get('error', err.get('user_message', ''))}")
    if confidence is not None:
        confidence_num = coerce_numeric_score(confidence)
        if confidence_num is not None and confidence_num < 70:
            risks.append(f"⚠ AI confidence is moderate ({confidence_num:.0f}%) — validate before production use")
    if not risks:
        risks.append("⚠ No critical risks flagged — continue monitoring drift and data quality in production")

    documentation = (output or {}).get("documentation") or {}
    business_rec = (output or {}).get("recommendation") or ""
    if isinstance(documentation, dict):
        sections = documentation.get("sections") or {}
        if isinstance(sections, dict):
            deploy_rec = sections.get("Deployment Recommendation")
            future = sections.get("Future Improvements")
            if deploy_rec:
                business_rec = f"{business_rec} {deploy_rec}".strip()
            elif future:
                business_rec = f"{business_rec} {future}".strip()
        if documentation.get("summary") and not business_rec:
            business_rec = str(documentation.get("summary"))
    session_recs = normalize_recommendations(st.session_state.get(SessionKeys.RECOMMENDATIONS))
    if not business_rec and session_recs:
        business_rec = " ".join(str(r) for r in session_recs[:2])
    if not business_rec:
        business_rec = "Run the autonomous AI scientist pipeline to generate a business recommendation."
    if selection_notes:
        business_rec = f"{business_rec} {' '.join(selection_notes[:2])}".strip()

    final_score = (output or {}).get("final_score")
    final_scores = (output or {}).get("final_scores") or {}
    if final_score is None and isinstance(final_scores, dict):
        final_score = final_scores.get("overall_score")

    risk_level = deploy.get("risk_level", "Unknown") if isinstance(deploy, dict) else "Unknown"
    deploy_key, deploy_label, deploy_color = _deployment_status(final_score, risk_level, business_rec)

    model_selection = None
    if isinstance(documentation, dict):
        sections = documentation.get("sections") or {}
        if isinstance(sections, dict):
            model_selection = sections.get("Model Selection")

    reasoning_summary = model_selection or ""
    if selection_notes and not reasoning_summary:
        reasoning_summary = selection_notes[0]
    if best_name and best_score is not None and not reasoning_summary:
        _, formatted = _format_metric(best_score, problem_type)
        reasoning_summary = (
            f"{best_name} achieved the highest {metric_label.lower()} ({formatted}) among evaluated candidates "
            f"for this {problem_type.lower()} task."
        )

    return {
        "pipeline_complete": has_autonomous_result(),
        "model_name": best_name or "—",
        "model_version": version or "—",
        "metric_label": metric_label,
        "metric_value": metric_value,
        "f1_display": f1_display,
        "problem_type": problem_type,
        "confidence": confidence,
        "final_score": final_score,
        "health_score": health.get("score"),
        "health_grade": health.get("grade"),
        "health_summary": health.get("summary"),
        "intelligence_score": intelligence_score,
        "missing_total": missing_total,
        "missing_pct": missing_pct,
        "outlier_note": outlier_note,
        "issues": issues,
        "leaderboard": leaderboard,
        "validation_lines": validation_lines,
        "overfitting_lines": overfitting_lines,
        "stability_lines": stability_lines,
        "explain_lines": explain_lines,
        "strengths": strengths,
        "risks": risks,
        "business_recommendation": business_rec,
        "reasoning_summary": reasoning_summary,
        "model_selection_notes": selection_notes,
        "deployment_key": deploy_key,
        "deployment_label": deploy_label,
        "deployment_color": deploy_color,
        "risk_level": risk_level,
        "trust_score": trust_score,
    }


def _render_confidence_gauge(confidence: Optional[float]) -> None:
    value = float(confidence) if confidence is not None else 0.0
    value = min(100.0, max(0.0, value))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 28, "color": PRIMARY_COLOR}},
            title={"text": "AI Confidence", "font": {"size": 14, "color": "#64748B"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": ACCENT_COLOR, "thickness": 0.28},
                "bgcolor": "rgba(255,255,255,0.6)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "rgba(239,68,68,0.15)"},
                    {"range": [50, 75], "color": "rgba(245,158,11,0.15)"},
                    {"range": [75, 100], "color": "rgba(16,185,129,0.18)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_chief_decision_panel(output: Optional[Dict[str, Any]], *, compact: bool = False) -> None:
    """Render the executive AI Chief Data Scientist Decision panel."""
    df = get_dataframe()
    data = build_chief_decision_data(output, df)

    if compact:
        st.markdown(
            f"""
            <div class="chief-decision-card chief-decision-compact">
                <div class="chief-section-label">Recommended Model</div>
                <div class="chief-model-name">{data['model_name']}</div>
                <div class="chief-model-meta">
                    <span><strong>{data['metric_label']}</strong> {data['metric_value']}</span>
                    <span><strong>Task</strong> {data['problem_type']}</span>
                </div>
                <div class="chief-deployment-status" style="color:{data['deployment_color']};margin-top:0.5rem;">
                    {data['deployment_label']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if data["business_recommendation"]:
            st.caption(data["business_recommendation"][:220] + ("…" if len(data["business_recommendation"]) > 220 else ""))
        with st.expander("Full executive report", expanded=False):
            render_chief_decision_panel(output, compact=False)
        return

    st.markdown(
        """
        <div class="chief-decision-shell">
            <div class="chief-decision-header">
                <div class="chief-decision-icon">🧠</div>
                <div>
                    <div class="chief-decision-kicker">Executive Report</div>
                    <div class="chief-decision-title">AI Chief Data Scientist Decision</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not data["pipeline_complete"]:
        st.info("Upload a dataset and run the autonomous pipeline to populate this executive decision report.")

    col_model, col_gauge = st.columns([1.4, 1])
    with col_model:
        st.markdown(
            f"""
            <div class="chief-decision-card">
                <div class="chief-section-label">Recommended Model</div>
                <div class="chief-model-name">{data['model_name']}</div>
                <div class="chief-model-meta">
                    <span><strong>Version</strong> {data['model_version']}</span>
                    <span><strong>{data['metric_label']}</strong> {data['metric_value']}</span>
                    <span><strong>F1</strong> {data['f1_display']}</span>
                    <span><strong>Task</strong> {data['problem_type']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_gauge:
        if data["confidence"] is not None:
            _render_confidence_gauge(data["confidence"])
        else:
            st.markdown(
                "<div class='chief-decision-card chief-muted'>Confidence score unavailable until model training completes.</div>",
                unsafe_allow_html=True,
            )

    if data["reasoning_summary"]:
        st.markdown(f"<div class='chief-summary'>{data['reasoning_summary']}</div>", unsafe_allow_html=True)

    with st.expander("AI Reasoning", expanded=False):
        st.markdown("**Model leaderboard**")
        if data["leaderboard"]:
            st.dataframe(pd.DataFrame(data["leaderboard"]), width="stretch", hide_index=True)
        else:
            st.caption("Leaderboard unavailable — complete AutoML training.")

        st.markdown("**Cross-validation**")
        if data["validation_lines"]:
            for line in data["validation_lines"]:
                st.write(f"- {line}")
        else:
            st.caption("Validation metrics not yet available from the pipeline.")

        st.markdown("**Overfitting checks**")
        for line in data["overfitting_lines"]:
            st.write(f"- {line}")

        st.markdown("**Training stability**")
        for line in data["stability_lines"]:
            st.write(f"- {line}")

        st.markdown("**Explainability**")
        for line in data["explain_lines"]:
            st.write(f"- {line}")

    col_health, col_strength = st.columns(2)
    with col_health:
        st.markdown(
            f"""
            <div class="chief-decision-card">
                <div class="chief-section-label">Dataset Health</div>
                <div class="chief-stat-row">
                    <span class="chief-stat">Health Score<br><strong>{data['health_score'] if data['health_score'] is not None else '—'}/100</strong></span>
                    <span class="chief-stat">Grade<br><strong>{data['health_grade'] or '—'}</strong></span>
                    <span class="chief-stat">Missing<br><strong>{data['missing_total']:,} ({data['missing_pct']:.1f}%)</strong></span>
                </div>
                <div class="chief-muted-text">{data['health_summary'] or 'Health summary unavailable.'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if data["outlier_note"]:
            st.caption(f"Outliers: {data['outlier_note']}")
        elif data["issues"]:
            issue_titles = [
                i.get("title", "") for i in data["issues"][:4]
                if isinstance(i, dict) and i.get("title") != "No Critical Issues"
            ]
            if issue_titles:
                st.caption("Data quality risks: " + "; ".join(issue_titles))

    with col_strength:
        st.markdown("<div class='chief-section-label'>Model Strengths</div>", unsafe_allow_html=True)
        for item in data["strengths"]:
            st.markdown(f"<div class='chief-strength'>{item}</div>", unsafe_allow_html=True)

    st.markdown("<div class='chief-section-label'>Risks</div>", unsafe_allow_html=True)
    for risk in data["risks"][:8]:
        st.markdown(f"<span class='chief-risk-badge'>{risk}</span>", unsafe_allow_html=True)

    st.markdown("<div class='chief-section-label'>Business Recommendation</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='chief-decision-card chief-business'><div class='chief-business-text'>{data['business_recommendation']}</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="chief-deployment-banner" style="border-color:{data['deployment_color']};">
            <div class="chief-section-label">Deployment Decision</div>
            <div class="chief-deployment-status" style="color:{data['deployment_color']};">{data['deployment_label']}</div>
            <div class="chief-muted-text">
                Overall score: {data['final_score'] if data['final_score'] is not None else '—'}/100 ·
                Risk level: {data['risk_level']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
