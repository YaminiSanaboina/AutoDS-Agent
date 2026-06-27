"""Enterprise dashboard views — data assembly and section renderers."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agents.eda_agent import correlation_heatmap, missing_values_chart
from config import ACCENT_COLOR, PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, WARNING_COLOR
from ui.chief_decision_panel import build_chief_decision_data, render_chief_decision_panel
from ui.components import (
    render_agent_terminal,
    render_executive_banner,
    render_leaderboard_table,
    section_header,
    primary_metric_label,
)
from ui.interactive_report_center import render_interactive_report_center
from ui.saas_components import (
    render_deployment_status_card,
    render_home_hero,
    render_live_execution_panel,
    render_slim_insight_panel,
)
from ui.step_detail_panels import render_step_detail
from utils.health_score import compute_health_score
from utils.recommendations_engine import generate_live_recommendations
from utils.faculty_labels import FACULTY_HELP
from utils.safe_checks import (
    coerce_numeric_score,
    coalesce_dict,
    coalesce_list,
    display_kpi_value,
    format_accuracy_display,
    format_score_display,
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


STAGE_ACTIVITY_MESSAGES = {
    "dataset_upload": "Dataset uploaded and validated",
    "dataset_intelligence": "Dataset analyzed",
    "data_cleaning": "Missing values handled",
    "eda": "Exploratory analysis complete",
    "feature_engineering": "Feature engineering complete",
    "automl": "Model training complete",
    "model_comparison": "Model comparison finished",
    "explainability": "Generating SHAP values",
    "ai_ethics_trust": "Ethics assessment complete",
    "self_improvement": "Self-improvement cycle complete",
    "deployment_readiness": "Deployment assessment complete",
    "monitoring": "Monitoring plan configured",
    "ai_decision": "Final AI decision generated",
    "prediction": "Prediction engine ready",
    "pdf_report": "Executive report generated",
}


from utils.stage_display import format_stage_display

_logger = logging.getLogger(__name__)


def _log_missing_dashboard_section(section: str, reason: str = "not available") -> None:
    """Log missing pipeline sections without interrupting UI rendering."""
    _logger.debug("[dashboard] Missing section '%s' (%s)", section, reason)


def _safe_output_dict(output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize pipeline output to a dict; fall back to session autonomous result."""
    if isinstance(output, dict) and output:
        return output
    if has_autonomous_result():
        try:
            cached = get_autonomous_result()
            if isinstance(cached, dict):
                return cached
        except Exception:
            _log_missing_dashboard_section("autonomous_result", "session read failed")
    return {}


def build_dashboard_context(output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble metrics for command bar, hero, and insight panel.

    Every pipeline section is optional. All variables are initialized before use so
    partial or missing pipeline output never raises UnboundLocalError / KeyError.
    """
    # --- Dataset & health (safe even with no upload) ---
    df = None
    try:
        df = get_dataframe() if has_dataset() else None
    except Exception:
        _log_missing_dashboard_section("dataset", "session dataframe unreadable")

    target_col = st.session_state.get(SessionKeys.TARGET_COLUMN)
    health: Any = get_session_health(df, target_column=target_col)

    if isinstance(health, dict):
        health_score = float(health.get("score", 0) or 0)
        health_grade = health.get("letter_grade") or health.get("grade", "Unknown")
    else:
        try:
            health_score = float(health) if health is not None else 0.0
        except (TypeError, ValueError):
            health_score = 0.0
        health_grade = "Unknown"

    # --- Session / pipeline execution state ---
    is_running = bool(st.session_state.get(SessionKeys.PIPELINE_RUNNING, False))
    progress = int(st.session_state.get(SessionKeys.PIPELINE_PROGRESS, 0) or 0)
    current_stage = st.session_state.get(SessionKeys.PIPELINE_CURRENT_STAGE) or "—"
    elapsed = float(st.session_state.get(SessionKeys.PIPELINE_ELAPSED_TIME, 0) or 0)
    session_id = st.session_state.get(SessionKeys.PIPELINE_EXECUTION_STATE_ID) or "—"

    # --- Initialize ALL pipeline-derived structures with safe defaults ---
    output_data = _safe_output_dict(output)
    model_results: Dict[str, Any] = {}
    automl_results: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {}
    detailed_metrics: Dict[str, Any] = {}
    deployment_readiness: Dict[str, Any] = {}
    trust_metrics: Dict[str, Any] = {}
    shap_results: Dict[str, Any] = {}
    feature_importance: Dict[str, Any] = {}
    analysis_profile: Dict[str, Any] = coalesce_dict(st.session_state.get(SessionKeys.ANALYSIS_PROFILE))
    validation: Dict[str, Any] = coalesce_dict(st.session_state.get(SessionKeys.VALIDATION_REPORT))

    best_model = "—"
    score_val: Optional[float] = None
    confidence_val: Optional[float] = None
    validation_score: Optional[float] = None
    fairness_score: Optional[float] = None
    model_readiness_score: Optional[float] = None
    reliability: Optional[float] = None
    deployment_label = "Pending"
    production_badge = "Awaiting Pipeline"
    production_class = "ads-badge-review"
    recommendations: List[str] = []
    score_display_cached: Optional[str] = None

    # === READ FROM EXECUTIVE_METRICS FIRST (Single Source of Truth) ===
    executive_metrics = st.session_state.get(SessionKeys.EXECUTIVE_METRICS) or {}
    if executive_metrics:
        # Use authoritative executive metrics for consistency
        best_model = display_kpi_value(executive_metrics.get("best_model"))
        score_val = resolve_canonical_accuracy(
            executive_metrics,
            coalesce_dict(safe_dict_get(output_data, "model_results")),
            best_model=executive_metrics.get("best_model"),
            session_best_score=st.session_state.get(SessionKeys.BEST_SCORE),
        )
        confidence_val = executive_metrics.get("trust_score") if executive_metrics.get("trust_score") is not None else executive_metrics.get("confidence_score")
        deployment_label = display_kpi_value(executive_metrics.get("deployment_status"), unavailable="Pending")
        if executive_metrics.get("accuracy_display"):
            score_display_cached = executive_metrics["accuracy_display"]
        else:
            score_display_cached = None
        executive_health = executive_metrics.get("health_score")
        if isinstance(executive_health, dict):
            health_score = float(executive_health.get("score", 0) or 0)
            health_grade = executive_health.get("letter_grade") or executive_health.get("grade", "Unknown")
        elif executive_health is not None:
            try:
                health_score = float(executive_health)
            except (TypeError, ValueError):
                health_score = health_score
            health_grade = health_grade
        deployment_readiness = coalesce_dict(safe_dict_get(executive_metrics, "deployment_readiness"))
        trust_metrics = coalesce_dict(executive_metrics)
        risk = executive_metrics.get("risk_level", "Unknown")
        recommendations = normalize_recommendations(safe_dict_get(executive_metrics.get("final_decision") or {}, "recommendation"))
        if not recommendations:
            recommendations = normalize_recommendations(st.session_state.get(SessionKeys.RECOMMENDATIONS))
        
        # Map risk level to badge
        if risk.lower() == "low" and deployment_label == "Production Ready":
            production_badge = "Production Ready"
            production_class = "ads-badge-ready"
        elif risk.lower() == "medium" or "Monitoring" in deployment_label:
            production_badge = "Needs Monitoring"
            production_class = "ads-badge-review"
        else:
            production_badge = "Not Ready"
            production_class = "ads-badge-caution"

    if output_data and not executive_metrics:
        model_results = coalesce_dict(safe_dict_get(output_data, "model_results"))
        automl_results = model_results
        if not model_results:
            _log_missing_dashboard_section("model_results")

        metrics = coalesce_dict(safe_dict_get(model_results, "metrics"))
        detailed_metrics = coalesce_dict(
            st.session_state.get(SessionKeys.MODEL_METRICS)
            or safe_dict_get(model_results, "detailed_metrics")
        )

        deployment_readiness = coalesce_dict(safe_dict_get(output_data, "deployment_readiness"))
        if not deployment_readiness:
            _log_missing_dashboard_section("deployment_readiness")

        trust_metrics = coalesce_dict(safe_dict_get(output_data, "ai_trust_results"))
        if not trust_metrics:
            trust_metrics = coalesce_dict(safe_dict_get(output_data, "ethics_report"))

        shap_results = coalesce_dict(safe_dict_get(output_data, "explainability_results"))
        if not shap_results:
            shap_results = coalesce_dict(safe_dict_get(output_data, "xai_results"))

        fi_raw = safe_dict_get(shap_results, "feature_importance")
        if isinstance(fi_raw, dict):
            feature_importance = fi_raw

        if not validation:
            validation = coalesce_dict(safe_dict_get(output_data, "validation"))

        best_model = (
            safe_dict_get(model_results, "best_model")
            or st.session_state.get(SessionKeys.BEST_MODEL_NAME)
            or "Unavailable"
        )

        if best_model != "Unavailable" and best_model in metrics:
            score_val = metrics.get(best_model)
        elif st.session_state.get(SessionKeys.BEST_SCORE) is not None:
            score_val = st.session_state.get(SessionKeys.BEST_SCORE)

        confidence_val = safe_dict_get(output_data, "final_ai_confidence_score")
        if confidence_val is None:
            confidence_val = st.session_state.get(SessionKeys.CONFIDENCE_SCORE)
        if confidence_val is None and score_val is not None:
            try:
                confidence_val = float(score_val) * 100 if float(score_val) <= 1 else float(score_val)
            except (TypeError, ValueError):
                _log_missing_dashboard_section("confidence", "score conversion failed")

        if confidence_val is None and best_model != "—":
            dm = coalesce_dict(detailed_metrics.get(best_model))
            if dm:
                try:
                    raw = dm.get("cv_score") or dm.get("accuracy") or dm.get("r2") or 0
                    confidence_val = float(raw) * 100
                except (TypeError, ValueError):
                    pass

        risk = safe_dict_get(deployment_readiness, "risk_level") or "Unknown"
        if risk == "Low":
            deployment_label = "Production Ready"
            production_badge = "Production Ready"
            production_class = "ads-badge-ready"
        elif risk == "Medium" or "Monitoring" in deployment_label:
            deployment_label = "Needs Monitoring"
            production_badge = "Review Required"
            production_class = "ads-badge-review"
        elif risk != "Unknown":
            deployment_label = "High Risk"
            production_badge = "Not Ready"
            production_class = "ads-badge-review"

        recommendations = normalize_recommendations(safe_dict_get(deployment_readiness, "recommendations"))
        if not recommendations:
            recommendations = normalize_recommendations(st.session_state.get(SessionKeys.RECOMMENDATIONS))
        if not recommendations and executive_metrics:
            recommendations = normalize_recommendations(safe_dict_get(executive_metrics.get("final_decision") or {}, "recommendation"))
        if not recommendations:
            recommendations = normalize_recommendations(safe_dict_get(output_data, "recommendation"))
        if not recommendations and is_present(df):
            try:
                recommendations = generate_live_recommendations(df, output_data)
            except Exception:
                _log_missing_dashboard_section("recommendations", "generation failed")
    elif is_present(df):
        _log_missing_dashboard_section("pipeline_output", "using dataset-only recommendations")
        try:
            recommendations = generate_live_recommendations(df, None)
        except Exception:
            recommendations = []

    # Composite readiness — variables always defined above; never branch-gated
    validation_score = validation.get("overall_score") if validation else None
    if validation_score is None:
        validation_score = st.session_state.get(SessionKeys.VALIDATION_SCORE)

    fairness_score = trust_metrics.get("fairness_score") if trust_metrics else None
    model_readiness_score = model_results.get("model_readiness_score") if model_results else None

    readiness_parts = [
        v for v in (validation_score, coerce_numeric_score(fairness_score), model_readiness_score)
        if v is not None
    ]
    if readiness_parts:
        try:
            composite = sum(float(v) for v in readiness_parts) / len(readiness_parts)
            if composite >= 80 and deployment_label == "Pending":
                deployment_label = f"{composite:.0f}/100 Ready"
        except (TypeError, ValueError):
            _log_missing_dashboard_section("readiness_composite", "numeric conversion failed")

    if is_running:
        ai_status = "Running"
        pipeline_status = "Executing"
    elif has_autonomous_result():
        ai_status = "Completed"
        pipeline_status = "Complete"
    elif has_dataset():
        ai_status = "Ready"
        pipeline_status = "Idle"
    else:
        ai_status = "Waiting"
        pipeline_status = "No Dataset"

    remaining = "—"
    if is_running and progress > 0 and progress < 100:
        try:
            est_total = elapsed / (progress / 100.0)
            remaining = f"{max(0, int(est_total - elapsed))}s"
        except ZeroDivisionError:
            remaining = "—"

    chief: Dict[str, Any] = {}
    try:
        chief = build_chief_decision_data(output_data or output, df) if (output_data or has_dataset()) else {}
    except Exception:
        _log_missing_dashboard_section("chief_decision", "build failed")

    score_display = (
        score_display_cached
        if executive_metrics and score_display_cached
        else format_accuracy_display(
            score_val,
            get_problem_type(output_data),
            unavailable="Unavailable",
        )
    )

    confidence_display = "Unavailable"
    if confidence_val is not None:
        try:
            confidence_display = f"{float(confidence_val):.0f}%"
            reliability = float(confidence_val)
        except (TypeError, ValueError):
            confidence_display = "Unavailable"

    final_report = coalesce_dict(safe_dict_get(output_data, "final_report")) if output_data else {}
    report_payload = coalesce_dict(safe_dict_get(final_report, "payload")) if final_report else {}
    if executive_metrics and executive_metrics.get("trust_score") is not None:
        trust_score_raw = executive_metrics.get("trust_score")
    elif report_payload and report_payload.get("trust_score") is not None:
        trust_score_raw = report_payload.get("trust_score")
    else:
        trust_score_raw = safe_dict_get(trust_metrics, "trust_score") if trust_metrics else None
        if trust_score_raw is None and trust_metrics:
            governance = coalesce_dict(safe_dict_get(trust_metrics, "ai_governance_score"))
            if isinstance(governance, dict):
                trust_score_raw = governance.get("score")
            else:
                trust_score_raw = governance
    trust_numeric = coerce_numeric_score(trust_score_raw)
    if trust_numeric is not None:
        trust_display = f"{trust_numeric:.0f}/100"
    elif is_running:
        trust_display = "Calculating…"
    else:
        trust_display = "Unavailable"

    return {
        "health_score": health_score,
        "health_grade": health_grade,
        "health_display": f"{health_score:.0f}/100 ({health_grade})",
        "dataset_status": "Loaded" if has_dataset() else "Not Loaded",
        "current_stage": format_stage_display(current_stage),
        "is_running": is_running,
        "progress": progress,
        "elapsed": elapsed,
        "session_id": session_id,
        "best_model": best_model,
        "score_val": score_val,
        "score_display": score_display,
        "confidence_display": confidence_display,
        "reliability_display": confidence_display,
        "trust_display": trust_display,
        "trust_score_raw": trust_score_raw,
        "deployment_label": deployment_label,
        "production_badge": production_badge,
        "production_class": production_class,
        "ai_status": ai_status,
        "pipeline_status": pipeline_status,
        "remaining": remaining,
        "recommendations": recommendations,
        "chief": chief,
        "timer_text": f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}",
        # Expose normalized sections for downstream panels (all safe defaults)
        "validation": validation,
        "validation_score": validation_score,
        "deployment_readiness": deployment_readiness,
        "model_metrics": metrics,
        "detailed_metrics": detailed_metrics,
        "reliability": reliability,
        "feature_importance": feature_importance,
        "analysis_profile": analysis_profile,
        "automl_results": automl_results,
        "shap_results": shap_results,
        "trust_metrics": trust_metrics,
    }


def build_terminal_lines() -> List[Tuple[str, str]]:
    """Build live agent activity terminal lines from pipeline state."""
    lines: List[Tuple[str, str]] = []
    statuses = st.session_state.get(SessionKeys.PIPELINE_STAGE_STATUSES, {})
    current = st.session_state.get(SessionKeys.PIPELINE_CURRENT_STAGE)
    is_running = st.session_state.get(SessionKeys.PIPELINE_RUNNING, False)

    for stage_key, message in STAGE_ACTIVITY_MESSAGES.items():
        status = statuses.get(stage_key)
        if status == "completed":
            lines.append((f"✓ {message}", "success"))
        elif status == "active" or (is_running and stage_key == current):
            lines.append((f"⟳ {message}…", "active"))
        elif has_autonomous_result():
            lines.append((f"✓ {message}", "success"))

    if not lines and has_dataset():
        lines.append(("○ Upload complete — ready to run pipeline", "pending"))
    if not lines:
        lines.append(("○ Waiting for dataset upload…", "pending"))
    return lines[-12:]


def build_leaderboard_rows(output: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build AutoML leaderboard rows from pipeline output."""
    from utils.model_ranking import rank_models_by_composite

    if not output:
        return []
    model_results = coalesce_dict(safe_dict_get(output, "model_results"))
    metrics = coalesce_dict(safe_dict_get(model_results, "metrics"))
    training_times = coalesce_dict(safe_dict_get(model_results, "training_times"))
    best = safe_dict_get(model_results, "best_model")
    problem_type = get_problem_type(output)
    detailed = coalesce_dict(safe_dict_get(model_results, "detailed_metrics"))

    rows = []
    ranked = rank_models_by_composite(detailed, problem_type, metrics)
    if ranked:
        for rank, (name, score) in enumerate(ranked, start=1):
            dm = coalesce_dict(detailed.get(name))
            if problem_type == "Classification":
                acc = f"{score * 100:.1f}%" if score <= 1 else f"{score:.4f}"
            else:
                acc = f"R² {score:.4f}" if score <= 1 else f"{score:.4f}"
            tv = training_times.get(name, dm.get("training_time", "—"))
            time_str = f"{tv:.2f}s" if isinstance(tv, (int, float)) else str(tv)
            rows.append({
                "rank": rank,
                "model": name,
                "accuracy": acc,
                "f1": "—",
                "time": time_str,
                "status": "✓ Best" if name == best else "",
            })
        return rows

    for rank, (name, score) in enumerate(sorted(metrics.items(), key=lambda x: x[1], reverse=True), start=1):
        dm = coalesce_dict(detailed.get(name))
        if problem_type == "Classification":
            acc = f"{score * 100:.1f}%" if score <= 1 else f"{score:.4f}"
        else:
            acc = f"R² {score:.4f}" if score <= 1 else f"{score:.4f}"
        tv = training_times.get(name, dm.get("training_time", "—"))
        time_str = f"{tv:.2f}s" if isinstance(tv, (int, float)) else str(tv)
        rows.append({
            "rank": rank,
            "model": name,
            "accuracy": acc,
            "f1": "—",
            "time": time_str,
            "status": "✓ Best" if name == best else "",
        })
    return rows


def render_timeline_html(
    steps: List[Dict[str, str]],
    get_status_fn: Callable[[Dict[str, str]], str],
) -> str:
    """Render enterprise timeline as HTML string."""
    parts = ['<div class="ads-timeline">']
    for item in steps:
        status = get_status_fn(item)
        css = {"COMPLETED": "done", "ACTIVE": "active"}.get(status, "")
        badge_cls = {"COMPLETED": "done", "ACTIVE": "active", "WAITING": "waiting"}.get(status, "waiting")
        icon = "✓" if status == "COMPLETED" else ("⟳" if status == "ACTIVE" else "○")
        duration = "Complete" if status == "COMPLETED" else ("Running…" if status == "ACTIVE" else "Waiting")
        subtitle = item.get("subtitle", "")
        parts.append(
            f"""
            <div class="ads-timeline-item {css}">
                <div class="ads-timeline-dot">{icon}</div>
                <div class="ads-timeline-card {css}">
                    <div class="ads-timeline-header">
                        <h4>{item['label']}</h4>
                        <span class="ads-step-badge {badge_cls}">{status}</span>
                    </div>
                    <p style="margin:6px 0 0;color:#64748B;font-size:0.86rem;line-height:1.5;">{subtitle}</p>
                    <div class="ads-timeline-meta"><span>Duration: {duration}</span><span>Status: {status.title()}</span></div>
                </div>
            </div>
            """
        )
    parts.append("</div>")
    return "".join(parts)


def _eda_ai_observations(df: pd.DataFrame, output: Optional[Dict[str, Any]]) -> List[str]:
    """Build EDA observations from pipeline insights or computed stats."""
    eda = coalesce_dict(safe_dict_get(output, "eda_results"))
    insights = normalize_recommendations(safe_dict_get(eda, "insights"))
    if not insights:
        insights = normalize_recommendations(safe_dict_get(eda, "ai_insights"))
    if insights:
        return insights[:8]

    observations: List[str] = []
    missing_cols = df.isnull().sum()
    worst = missing_cols[missing_cols > 0]
    if not worst.empty:
        col = worst.idxmax()
        observations.append(f"Column '{col}' has the highest missing rate ({worst.max()} values).")
    numerics = df.select_dtypes(include="number").columns.tolist()
    if len(numerics) >= 2:
        corr = df[numerics].corr(numeric_only=True).abs()
        np.fill_diagonal(corr.values, 0)
        if not corr.empty:
            stacked = corr.stack()
            if not stacked.empty:
                pair = stacked.idxmax()
                observations.append(
                    f"Strongest correlation: {pair[0]} vs {pair[1]} ({stacked.max():.2f})."
                )
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        col = cat_cols[0]
        observations.append(f"'{col}' has {df[col].nunique()} unique categories.")
    if not observations:
        observations.append("Dataset structure looks consistent — review distributions and target balance.")
    return observations


def render_eda_dashboard(output: Optional[Dict[str, Any]], df: Optional[pd.DataFrame]) -> None:
    """Professional EDA Lab with tabbed sections and Plotly visualizations."""
    section_header("EDA Lab", "Exploratory data analysis — overview, quality, correlations, and target behavior.")
    if not is_present(df):
        return

    target_col = st.session_state.get(SessionKeys.TARGET_COLUMN)
    health = get_session_health(df, target_column=target_col)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Health Score", f"{health['score']:.0f}/100", health.get("letter_grade", ""))
    with c2:
        st.metric("Rows", f"{df.shape[0]:,}")
    with c3:
        st.metric("Columns", df.shape[1])
    with c4:
        st.metric("Missing Cells", f"{int(df.isnull().sum().sum()):,}")

    tabs = st.tabs([
        "Overview",
        "Missing Values",
        "Correlation",
        "Distribution",
        "Outliers",
        "Categorical",
        "Target Analysis",
    ])

    numerics = df.select_dtypes(include="number").columns.tolist()
    categoricals = df.select_dtypes(include=["object", "category"]).columns.tolist()

    with tabs[0]:
        st.dataframe(df.head(20), width="stretch")
        st.markdown("**Column dtypes**")
        dtype_df = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values})
        st.dataframe(dtype_df, width="stretch", hide_index=True)
        for rec in health.get("recommendations", [])[:5]:
            st.markdown(f"- {rec}")

    with tabs[1]:
        st.plotly_chart(missing_values_chart(df), width="stretch")
        miss = df.isnull().sum().sort_values(ascending=False)
        miss = miss[miss > 0]
        if not miss.empty:
            st.dataframe(
                pd.DataFrame({"Column": miss.index, "Missing": miss.values, "Pct": (miss / len(df) * 100).round(2)}),
                width="stretch",
                hide_index=True,
            )
        else:
            st.success("No missing values detected.")

    with tabs[2]:
        if len(numerics) >= 2:
            st.plotly_chart(correlation_heatmap(df), width="stretch")
            sc1, sc2 = st.columns(2)
            with sc1:
                x_col = st.selectbox("X axis", numerics, key="eda_scatter_x_dashboard")
            with sc2:
                y_col = st.selectbox("Y axis", numerics, index=min(1, len(numerics) - 1), key="eda_scatter_y_dashboard")
            color_col = st.selectbox("Color (optional)", ["None"] + categoricals, key="eda_scatter_color")
            scatter_kwargs = {"x": x_col, "y": y_col, "title": f"{x_col} vs {y_col}"}
            if color_col != "None":
                scatter_kwargs["color"] = color_col
            fig = px.scatter(df, **scatter_kwargs, color_discrete_sequence=[ACCENT_COLOR])
            fig.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Need at least two numeric columns for correlation analysis.")

    with tabs[3]:
        if numerics:
            selected = st.selectbox("Feature", numerics, key="eda_dash_feature")
            fig_h = px.histogram(df, x=selected, nbins=30, title=f"Histogram: {selected}", color_discrete_sequence=[ACCENT_COLOR])
            fig_h.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_h, width="stretch")
            fig_v = px.violin(df, y=selected, box=True, points="outliers", title=f"Violin: {selected}", color_discrete_sequence=[SECONDARY_COLOR])
            fig_v.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_v, width="stretch")
        else:
            st.info("No numeric columns available for distribution charts.")

    with tabs[4]:
        if numerics:
            outlier_col = st.selectbox("Numeric feature", numerics, key="eda_outlier_col")
            fig_b = px.box(df, y=outlier_col, title=f"Outlier detection: {outlier_col}", color_discrete_sequence=[WARNING_COLOR])
            fig_b.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_b, width="stretch")
            q1, q3 = df[outlier_col].quantile(0.25), df[outlier_col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n_out = int(((df[outlier_col] < lower) | (df[outlier_col] > upper)).sum())
                st.metric("IQR Outliers", n_out)
        else:
            st.info("No numeric columns for outlier detection.")

    with tabs[5]:
        if categoricals:
            cat_col = st.selectbox("Categorical feature", categoricals, key="eda_cat_col")
            counts = df[cat_col].value_counts().head(12)
            fig_pie = px.pie(names=counts.index.astype(str), values=counts.values, title=f"Category mix: {cat_col}")
            fig_pie.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_pie, width="stretch")
            fig_bar = px.bar(x=counts.index.astype(str), y=counts.values, title=f"Top categories: {cat_col}", color_discrete_sequence=[PRIMARY_COLOR])
            fig_bar.update_layout(template="plotly_white", height=380)
            st.plotly_chart(fig_bar, width="stretch")
        else:
            st.info("No categorical columns detected.")

    with tabs[6]:
        candidates = [c for c in df.columns if c != target_col]
        tgt = target_col or st.selectbox("Select target column", df.columns, key="eda_target_pick")
        if tgt and tgt in df.columns:
            if pd.api.types.is_numeric_dtype(df[tgt]) and df[tgt].nunique() > 15:
                fig_t = px.histogram(df, x=tgt, nbins=30, title=f"Target distribution: {tgt}")
            else:
                vc = df[tgt].value_counts()
                fig_t = px.bar(x=vc.index.astype(str), y=vc.values, title=f"Target balance: {tgt}", color_discrete_sequence=[SUCCESS_COLOR])
            fig_t.update_layout(template="plotly_white", height=400)
            st.plotly_chart(fig_t, width="stretch")
            if numerics:
                feat = st.selectbox("Feature vs target", [c for c in numerics if c != tgt], key="eda_target_feat")
                if pd.api.types.is_numeric_dtype(df[tgt]) and df[tgt].nunique() > 10:
                    fig_st = px.scatter(df, x=feat, y=tgt, title=f"{feat} vs {tgt}", opacity=0.6)
                else:
                    fig_st = px.box(df, x=tgt, y=feat, title=f"{feat} by {tgt}")
                fig_st.update_layout(template="plotly_white", height=400)
                st.plotly_chart(fig_st, width="stretch")

    observations = _eda_ai_observations(df, output)
    if observations:
        st.markdown("**AI Observations**")
        for obs in observations:
            st.markdown(f"- {obs}")


def render_explainability_dashboard(output: Optional[Dict[str, Any]]) -> None:
    """Feature impact analysis dashboard with faculty-friendly explanations."""
    section_header(
        "Feature Impact Analysis",
        FACULTY_HELP["feature_impact"],
    )
    explainability = coalesce_dict(safe_dict_get(output, "explainability_results"))
    if not explainability:
        explainability = coalesce_dict(safe_dict_get(output, "xai_results"))

    fi_raw = st.session_state.get(SessionKeys.SHAP_IMPORTANCE)
    if not is_present(fi_raw):
        fi_raw = st.session_state.get(SessionKeys.FEATURE_IMPORTANCE)
    if not is_present(fi_raw):
        fi_raw = safe_dict_get(explainability, "feature_importance")
    fi_records = normalize_feature_importance(fi_raw)
    shap_ready = bool(st.session_state.get(SessionKeys.SHAP_COMPUTED))

    if not fi_records:
        if shap_ready:
            st.warning(
                "SHAP completed, but feature importance could not be parsed for display. "
                "Check that importance values are numeric and feature names are present."
            )
        elif is_present(fi_raw):
            st.warning("Feature importance data is present but could not be charted.")
        else:
            st.info("Explainability results appear after the SHAP stage completes.")
        return

    ranked = sorted(fi_records, key=lambda x: abs(x["importance"]), reverse=True)[:15]
    fig = go.Figure(
        go.Bar(
            x=[item["importance"] for item in ranked],
            y=[item["feature"] for item in ranked],
            orientation="h",
            marker_color=[SUCCESS_COLOR if item["importance"] >= 0 else WARNING_COLOR for item in ranked],
        )
    )
    fig.update_layout(title="Feature Importance (SHAP)", template="plotly_white", height=300)
    st.plotly_chart(fig, width="stretch")

    with st.expander("Feature drivers & ranking", expanded=False):
        pos = [item for item in ranked if item["importance"] >= 0][:5]
        neg = [item for item in ranked if item["importance"] < 0][:5]
        col_p, col_n = st.columns(2)
        with col_p:
            st.markdown("**Positive Drivers**")
            for item in pos:
                strength = "strongly" if abs(item["importance"]) > 0.15 else "moderately"
                st.markdown(
                    f"- **{item['feature']}**: Higher values {strength} increase the likelihood of a positive prediction."
                )
        with col_n:
            st.markdown("**Negative Drivers**")
            for item in (neg or ranked[:5]):
                st.markdown(
                    f"- **{item['feature']}**: Reduces the predicted outcome (impact {item['importance']:.4f})."
                )
        for idx, item in enumerate(ranked[:10], start=1):
            st.markdown(f"{idx}. `{item['feature']}` — {item['importance']:.4f}")

    explanation = safe_dict_get(explainability, "explanation") or safe_dict_get(explainability, "summary")
    if explanation:
        with st.expander("Natural language explanation", expanded=False):
            st.markdown(f'<div class="ads-glass-card">{explanation}</div>', unsafe_allow_html=True)


def render_automl_studio(output: Optional[Dict[str, Any]]) -> None:
    """Model training center with leaderboard, intelligence panel, and comparison chart."""
    section_header("Model Training Center", FACULTY_HELP["model_training"])
    rows = build_leaderboard_rows(output)
    best = None
    model_results = coalesce_dict(safe_dict_get(output, "model_results")) if output else {}
    if output:
        best = safe_dict_get(model_results, "best_model")
    render_leaderboard_table(rows, best_model=best)

    detailed = coalesce_dict(
        st.session_state.get(SessionKeys.MODEL_METRICS)
        or safe_dict_get(model_results, "detailed_metrics")
    )
    best_name = st.session_state.get(SessionKeys.BEST_MODEL_NAME) or best
    metrics_map = st.session_state.get(SessionKeys.RESULTS) or safe_dict_get(model_results, "metrics") or {}

    if best_name and best_name in detailed:
        dm = detailed[best_name]
        cv = dm.get("cv_score")
        with st.expander("Advanced model metrics", expanded=False):
            if cv is not None:
                metric_label = "Accuracy" if str(problem_type).lower() == "classification" else "R²"
                cv_display = f"{cv * 100:.1f}%" if cv <= 1 else f"{cv:.4f}"
                st.metric(f"{metric_label} (Cross-Validated)", cv_display)
            if str(problem_type).lower() == "regression":
                r2 = dm.get("r2")
                rmse = dm.get("rmse")
                mae = dm.get("mae")
                if r2 is not None:
                    r2_display = f"{r2:.4f}"
                    st.metric("R² Score", r2_display)
                if rmse is not None:
                    st.metric("RMSE", f"{rmse:.4f}")
                if mae is not None:
                    st.metric("MAE", f"{mae:.4f}")

        with st.expander("Model selection explanation", expanded=False):
            render_model_selection_explanation(output)

    if output and metrics_map:
        fig = go.Figure(go.Bar(x=list(metrics_map.keys()), y=list(metrics_map.values()), marker_color=ACCENT_COLOR))
        fig.update_layout(title="Model Comparison", template="plotly_white", height=280)
        st.plotly_chart(fig, width="stretch")
    elif not rows:
        st.info("Run Analysis from the Dashboard or Dataset Hub to train and compare models.")


def render_decision_center(output: Optional[Dict[str, Any]], *, compact: bool = False) -> None:
    """Executive AI decision dashboard."""
    if not compact:
        section_header("Final AI Decision Center", "Chief Data Scientist executive report and deployment recommendation.")
    if not has_autonomous_result() and not output:
        st.info("Complete the pipeline to unlock the executive decision center.")
        return

    df = get_dataframe() if has_dataset() else None
    data = build_chief_decision_data(output, df)
    if not compact:
        banner_cls = {"ready": "", "monitor": "warning", "not_ready": "danger"}.get(data.get("deployment_key", ""), "")
        render_executive_banner(data.get("deployment_label", "Pending"), banner_cls)
    render_chief_decision_panel(output, compact=compact)


def render_reports_center(output: Optional[Dict[str, Any]]) -> None:
    """Single organized AI report with six tabs and exports."""
    section_header("AI Report Center", "Executive summary, analysis, models, explainability, trust, and final decision.")
    render_interactive_report_center(output)


def render_stage_view(
    view_key: str,
    output: Optional[Dict[str, Any]],
    *,
    open_panel: Callable,
    close_panel: Callable,
    render_stat_grid: Callable,
    render_stage_output: Callable,
) -> None:
    """Render a pipeline stage detail view in the workspace."""
    stage_map = {
        "ai_cleaning": "data_cleaning",
        "ethics": "ai_ethics_trust",
        "deployment": "deployment_readiness",
    }
    step_key = stage_map.get(view_key, view_key)
    stage_results = st.session_state.get(SessionKeys.PIPELINE_STAGE_RESULTS, {})
    backend_key = step_key
    render_step_detail(
        step_key,
        output,
        stage_results,
        backend_key=backend_key,
        open_panel=open_panel,
        close_panel=close_panel,
        render_stat_grid=render_stat_grid,
        render_stage_output=render_stage_output,
    )


def render_right_panel(ctx: Dict[str, Any], *, active_view: str = "home") -> None:
    """Render insight panel on Assistant view only."""
    if active_view != "ai_assistant":
        return
    render_slim_insight_panel(
        ai_status=ctx["ai_status"],
        current_task=ctx["current_stage"],
        progress_pct=ctx["progress"],
        recommendations=ctx["recommendations"],
    )
    with st.expander("Live agent activity", expanded=False):
        render_agent_terminal(build_terminal_lines())


def render_model_selection_explanation(output: Optional[Dict[str, Any]]) -> None:
    """Faculty-friendly explanation of why the best model was selected."""
    if not output:
        return
    model_results = coalesce_dict(safe_dict_get(output, "model_results"))
    explanation = coalesce_dict(safe_dict_get(model_results, "model_selection_explanation"))
    if not explanation:
        st.info("Run the autonomous pipeline to generate the AI model selection explanation.")
        return

    section_header("AI Model Selection Explanation", "Why the platform chose this model over the alternatives.")
    st.markdown(f"**Selected Model:** {explanation.get('selected_model', '—')}")
    st.markdown("**Why chosen:**")
    for reason in coalesce_list(explanation.get("why_chosen")):
        st.markdown(f"- {reason}")

    for alt in coalesce_list(explanation.get("alternatives"))[:6]:
        if not isinstance(alt, dict):
            continue
        st.markdown(f"**Why {alt.get('model', 'Alternative')} was not chosen:**")
        for reason in coalesce_list(alt.get("reasons")):
            st.markdown(f"- {reason}")


def render_trust_fairness_dashboard(output: Optional[Dict[str, Any]]) -> None:
    """Trust AI module — reliability, data quality, bias, explainability, and risk scores."""
    section_header("Trust Center", FACULTY_HELP["trust_fairness"])
    executive_metrics = st.session_state.get(SessionKeys.EXECUTIVE_METRICS) or {}
    if executive_metrics:
        ethics = coalesce_dict(executive_metrics)
        deploy = coalesce_dict(safe_dict_get(executive_metrics, "deployment_readiness")) or coalesce_dict(executive_metrics.get("deployment_decision") or {})
    else:
        ethics = coalesce_dict(safe_dict_get(output, "ai_trust_results"))
        if not ethics:
            ethics = coalesce_dict(safe_dict_get(output, "ethics_report"))
        deploy = coalesce_dict(safe_dict_get(output, "deployment_readiness")) if output else {}

    df = get_dataframe() if has_dataset() else None
    target_col = st.session_state.get(SessionKeys.TARGET_COLUMN)
    health = get_session_health(df, target_column=target_col) if is_present(df) else {}
    if isinstance(health, dict):
        data_quality = float(health.get("score", 0) or 0)
    else:
        try:
            data_quality = float(health) if health is not None else 0.0
        except (TypeError, ValueError):
            data_quality = 0.0

    detailed = coalesce_dict(st.session_state.get(SessionKeys.MODEL_METRICS) or {})
    best_name = st.session_state.get(SessionKeys.BEST_MODEL_NAME)
    model_reliability = None
    if best_name and best_name in detailed:
        dm = detailed[best_name]
        parts = [dm.get(k) for k in ("accuracy", "precision", "recall", "f1", "r2", "cv_score") if dm.get(k) is not None]
        if parts:
            model_reliability = sum(float(p) if float(p) <= 1 else float(p) / 100 for p in parts) / len(parts) * 100

    explainability = coalesce_dict(safe_dict_get(output, "explainability_results"))
    fi = normalize_feature_importance(safe_dict_get(explainability, "feature_importance"))
    if not fi:
        fi = normalize_feature_importance(st.session_state.get(SessionKeys.SHAP_IMPORTANCE))
    explain_score = min(100.0, 40 + len(fi) * 4) if fi else (100.0 if st.session_state.get(SessionKeys.SHAP_COMPUTED) else None)

    trust_score = safe_dict_get(ethics, "trust_score")
    fairness_score = safe_dict_get(ethics, "fairness_score")
    governance = coalesce_dict(safe_dict_get(ethics, "ai_governance_score"))
    if trust_score is None and governance:
        trust_score = governance.get("score")
    if trust_score is None and model_reliability is not None:
        trust_score = round((model_reliability + data_quality) / 2, 1)

    risk_raw = safe_dict_get(deploy, "risk_level") or safe_dict_get(executive_metrics, "risk_level") or "Unknown"
    risk_map = {"Low": 15, "Medium": 45, "High": 75, "Critical": 90}
    risk_score = risk_map.get(str(risk_raw), 50)
    readiness_val = safe_dict_get(deploy, "readiness_score")
    if readiness_val is None:
        readiness_val = safe_dict_get(executive_metrics, "confidence_score")
    if readiness_val is not None:
        try:
            risk_score = max(0, min(100, 100 - float(readiness_val)))
        except (TypeError, ValueError):
            pass

    if not ethics and not has_autonomous_result() and not is_present(df):
        st.info("Upload a dataset and run analysis to generate the AI Trust Report.")
        return

    st.markdown("### Trust AI Report")
    render_kpi_strip(
        [
            {
                "label": "Model Reliability",
                "value": f"{coerce_numeric_score(model_reliability):.0f}/100" if coerce_numeric_score(model_reliability) is not None else format_score_display(model_reliability),
                "icon": "🎯",
            },
            {
                "label": "Data Quality",
                "value": f"{data_quality:.0f}/100",
                "icon": "📊",
            },
            {
                "label": "Bias / Fairness",
                "value": format_score_display(fairness_score, unavailable="Not Evaluated"),
                "icon": "⚖️",
            },
            {
                "label": "Explainability",
                "value": f"{explain_score:.0f}/100" if explain_score is not None else "Not Evaluated",
                "icon": "🔍",
            },
            {
                "label": "Risk Score",
                "value": f"{risk_score:.0f}/100",
                "icon": "⚠️",
            },
        ],
        columns=5,
    )

    with st.expander("Trust gauges & governance details", expanded=False):
        c1, c2 = st.columns(2)
        for col, label, value in (
            (c1, "Overall Trust Score", trust_score),
            (c2, "Fairness Score", fairness_score),
        ):
            with col:
                numeric = coerce_numeric_score(value)
                if numeric is not None:
                    fig = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=numeric,
                            number={"suffix": "/100"},
                            title={"text": label},
                            gauge={"axis": {"range": [0, 100]}, "bar": {"color": PRIMARY_COLOR}},
                        )
                    )
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=10), template="plotly_white")
                    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
                else:
                    st.metric(label, format_score_display(value, unavailable="Not Evaluated"))

        bias = normalize_recommendations(safe_dict_get(ethics, "bias_concerns"))
        privacy = normalize_recommendations(safe_dict_get(ethics, "privacy_concerns"))
        compliance = normalize_recommendations(safe_dict_get(ethics, "compliance_recommendations"))
        if not compliance:
            compliance = normalize_recommendations(safe_dict_get(ethics, "recommendations"))

        cards = st.columns(3)
        with cards[0]:
            st.markdown("**Bias Risk**")
            for item in (bias or ["No major bias concerns detected."])[:5]:
                st.markdown(f"- {item}")
        with cards[1]:
            st.markdown("**Privacy Assessment**")
            for item in (privacy or ["No sensitive identifiers flagged."])[:5]:
                st.markdown(f"- {item}")
        with cards[2]:
            st.markdown("**Compliance Assessment**")
            for item in (compliance or ["Standard ML governance checks applied."])[:5]:
                st.markdown(f"- {item}")

    summary = safe_dict_get(ethics, "executive_summary")
    if summary:
        with st.expander("Executive summary", expanded=False):
            st.markdown(
                f"<div class='report-insight-box'><strong>Executive Summary</strong><br>{summary}</div>",
                unsafe_allow_html=True,
            )


def render_home_page(
    output: Optional[Dict[str, Any]],
    *,
    upload_fn: Callable,
    run_fn: Callable,
) -> None:
    """Simplified home dashboard: dataset, quality, model, trust, run, top models."""
    ctx = build_dashboard_context(output)
    is_running = bool(ctx.get("is_running"))
    df = get_dataframe() if has_dataset() else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Dataset", get_dataset_name() if has_dataset() else "—")
        if df is not None:
            st.caption(f"{df.shape[0]:,} rows · {df.shape[1]} columns")
    with c2:
        st.metric("Problem Type", get_problem_type(output))
    with c3:
        st.metric("Health Score", ctx["health_display"].split(" ")[0] if ctx.get("health_display") else "—")
    with c4:
        st.metric("Best Model", str(ctx.get("best_model", "—")))

    c5, c6, c7 = st.columns(3)
    with c5:
        metric_label = primary_metric_label(get_problem_type(output))
        st.metric(metric_label, ctx.get("score_display", "Unavailable"))
    with c6:
        st.metric("Trust Score", ctx.get("trust_display", "—"))
    with c7:
        st.metric("Deployment", ctx.get("deployment_label", "Pending"))

    with st.expander("Dataset upload", expanded=not has_dataset()):
        upload_fn()

    if is_running:
        render_live_execution_panel(
            progress_pct=int(ctx.get("progress", 0) or 0),
            current_stage=str(st.session_state.get(SessionKeys.PIPELINE_CURRENT_STAGE) or "initializing"),
            stage_statuses=st.session_state.get(SessionKeys.PIPELINE_STAGE_STATUSES, {}),
            completed_stages=st.session_state.get(SessionKeys.PIPELINE_COMPLETED_STAGES, []),
            is_running=True,
        )
        return

    run_fn()

    if has_autonomous_result():
        section_header("Top 5 Models")
        rows = build_leaderboard_rows(output)
        best = safe_dict_get(coalesce_dict(safe_dict_get(output, "model_results")), "best_model") if output else None
        render_leaderboard_table(rows[:5], best_model=best, compact=True)
    elif has_dataset():
        st.caption("Click **Run Autonomous Analysis** to train models and generate your report.")


