"""Shared SaaS UI primitives — compact cards, steppers, KPI strips, and structured data views."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st

_PIPELINE_PHASES: List[Dict[str, Any]] = [
    {"id": "dataset", "label": "Dataset", "icon": "📁", "steps": ["dataset_upload", "dataset_intelligence"]},
    {"id": "prepare", "label": "Prepare", "icon": "🧹", "steps": ["data_cleaning", "eda", "feature_engineering"]},
    {"id": "train", "label": "Train", "icon": "🏆", "steps": ["automl", "model_comparison"]},
    {"id": "explain", "label": "Explain", "icon": "🔍", "steps": ["explainability"]},
    {"id": "govern", "label": "Govern", "icon": "🛡️", "steps": ["ai_ethics_trust", "self_improvement", "deployment_readiness", "monitoring"]},
    {"id": "deliver", "label": "Deliver", "icon": "📄", "steps": ["ai_decision", "prediction", "pdf_report"]},
]

_STATUS_ICON = {"COMPLETED": "✓", "ACTIVE": "●", "WAITING": "○"}


def _format_display(val: Any, *, max_len: int = 72) -> str:
    if val is None:
        return "—"
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        if abs(val) <= 1:
            return f"{val:.4f}"
        return f"{val:.2f}"
    if isinstance(val, int):
        return f"{val:,}"
    if isinstance(val, str):
        text = val.strip()
        return text[:max_len] + ("…" if len(text) > max_len else "")
    if isinstance(val, (list, tuple)):
        return f"{len(val)} items"
    if isinstance(val, dict):
        return f"{len(val)} fields"
    text = str(val)
    return text[:max_len] + ("…" if len(text) > max_len else "")


def render_metric_card(
    label: str,
    value: str,
    *,
    icon: str = "",
    help_text: str = "",
) -> None:
    icon_html = f'<span class="ads-kpi-icon">{icon}</span>' if icon else ""
    help_html = f'<div class="ads-kpi-help">{help_text}</div>' if help_text else ""
    st.markdown(
        f"""
        <div class="ads-kpi-card">
            {icon_html}
            <div class="ads-kpi-label">{label}</div>
            <div class="ads-kpi-value">{value}</div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_strip(items: List[Dict[str, str]], *, columns: int = 4) -> None:
    if not items:
        return
    cols = st.columns(min(columns, len(items)))
    for idx, item in enumerate(items):
        with cols[idx % len(cols)]:
            render_metric_card(
                str(item.get("label", "")),
                str(item.get("value", "—")),
                icon=str(item.get("icon", "")),
                help_text=str(item.get("help", "")),
            )


EXECUTION_DISPLAY_STAGES: List[Dict[str, Any]] = [
    {"label": "Dataset Understanding", "keys": ["dataset_upload", "dataset_intelligence", "initializing"]},
    {"label": "Data Cleaning", "keys": ["data_cleaning"]},
    {"label": "EDA", "keys": ["eda"]},
    {"label": "Feature Engineering", "keys": ["feature_engineering"]},
    {"label": "Training", "keys": ["automl"]},
    {"label": "Hyperparameter Optimization", "keys": ["automl"]},
    {"label": "Model Selection", "keys": ["model_comparison"]},
    {"label": "SHAP", "keys": ["explainability"]},
    {"label": "Trust Analysis", "keys": ["ai_ethics_trust", "self_improvement", "deployment_readiness", "monitoring"]},
    {"label": "Report Generation", "keys": ["ai_decision", "prediction", "pdf_report"]},
]


def render_home_hero(
    *,
    health: str,
    best_model: str,
    trust_score: str,
    deployment: str,
) -> None:
    st.markdown(
        f"""
        <div class="ads-hero ads-hero-compact">
            <div class="ads-hero-kicker">Autonomous Data Scientist</div>
            <h1>AutoDS Command Center</h1>
            <p>Upload a dataset, run autonomous analysis, and receive an executive AI decision report.</p>
            <div class="ads-hero-metrics">
                <div class="ads-metric-card">
                    <div class="ads-metric-label">Dataset Health</div>
                    <div class="ads-metric-value">{health}</div>
                </div>
                <div class="ads-metric-card">
                    <div class="ads-metric-label">Best Model</div>
                    <div class="ads-metric-value">{best_model}</div>
                </div>
                <div class="ads-metric-card">
                    <div class="ads-metric-label">Trust Score</div>
                    <div class="ads-metric-value">{trust_score}</div>
                </div>
                <div class="ads-metric-card">
                    <div class="ads-metric-label">Deployment Status</div>
                    <div class="ads-metric-value">{deployment}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compact_hero(
    *,
    title: str,
    subtitle: str,
    health: str,
    confidence: str,
    best_model: str,
    deployment: str,
) -> None:
    render_home_hero(
        health=health,
        best_model=best_model,
        trust_score=confidence,
        deployment=deployment,
    )


def _stage_display_status(
    stage_keys: List[str],
    *,
    stage_statuses: Dict[str, str],
    completed: List[str],
    current: Optional[str],
    is_running: bool,
) -> str:
    for key in stage_keys:
        if stage_statuses.get(key) == "active" or (is_running and key == current):
            return "ACTIVE"
    for key in stage_keys:
        if stage_statuses.get(key) == "completed" or key in completed:
            return "COMPLETED"
    return "WAITING"


def render_live_execution_panel(
    *,
    progress_pct: int,
    current_stage: str,
    stage_statuses: Dict[str, str],
    completed_stages: List[str],
    is_running: bool,
) -> None:
    """Live AI execution view with status indicators for each major phase."""
    st.markdown('<div class="ads-execution-panel">', unsafe_allow_html=True)
    st.markdown("#### Autonomous Analysis in Progress")
    st.progress(min(max(progress_pct, 0), 100) / 100.0)
    st.caption(f"Current stage: {current_stage.replace('_', ' ').title()} · {progress_pct}% complete")

    for item in EXECUTION_DISPLAY_STAGES:
        status = _stage_display_status(
            item["keys"],
            stage_statuses=stage_statuses,
            completed=completed_stages,
            current=current_stage,
            is_running=is_running,
        )
        icon = _STATUS_ICON.get(status, "○")
        cls = {"COMPLETED": "ads-exec-done", "ACTIVE": "ads-exec-active", "WAITING": "ads-exec-wait"}[status]
        pulse = " ads-exec-pulse" if status == "ACTIVE" else ""
        st.markdown(
            f"""
            <div class="ads-exec-row {cls}{pulse}">
                <span class="ads-exec-icon">{icon}</span>
                <span class="ads-exec-label">{item['label']}</span>
                <span class="ads-exec-badge">{status.replace('_', ' ').title()}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_deployment_status_card(label: str, *, status_key: str = "review") -> None:
    """Large executive deployment status card for Final Decision tab."""
    normalized = (label or "").lower()
    if "production ready" in normalized or status_key == "ready":
        css, title = "ads-status-ready", "PRODUCTION READY"
    else:
        css, title = "ads-status-review", "NEEDS REVIEW"
    st.markdown(
        f"""
        <div class="ads-deployment-status-card {css}">
            <div class="ads-deployment-status-title">{title}</div>
            <div class="ads-deployment-status-sub">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_structure_as_cards(
    val: Any,
    *,
    title: Optional[str] = None,
    depth: int = 0,
    max_items: int = 12,
) -> None:
    """Render dicts, lists, and primitives as professional metric cards instead of raw dumps."""
    if val is None:
        st.info("No results available.")
        return

    if isinstance(val, pd.DataFrame):
        if val.empty:
            st.info("Empty table.")
        else:
            st.dataframe(val.head(50), width="stretch", hide_index=True)
        return

    if isinstance(val, dict):
        if not val:
            st.info("No results available.")
            return
        if title:
            st.markdown(f"**{title}**")
        items = list(val.items())[:max_items]
        cols = st.columns(min(3, len(items)))
        for idx, (key, inner) in enumerate(items):
            with cols[idx % len(cols)]:
                if isinstance(inner, (dict, list)) and depth < 2:
                    render_metric_card(str(key), _format_display(inner))
                else:
                    render_metric_card(str(key), _format_display(inner))
        nested = {k: v for k, v in val.items() if isinstance(v, (dict, list)) and v}
        if nested and depth < 2:
            with st.expander("Advanced structured details", expanded=False):
                for key, inner in list(nested.items())[:8]:
                    render_structure_as_cards(inner, title=str(key), depth=depth + 1, max_items=8)
        return

    if isinstance(val, list):
        if not val:
            st.info("No results available.")
            return
        if all(isinstance(item, dict) for item in val[:5]):
            try:
                st.dataframe(pd.DataFrame(val), width="stretch", hide_index=True)
                return
            except Exception:
                pass
        pseudo = {f"Item {i + 1}": item for i, item in enumerate(val[:max_items])}
        render_structure_as_cards(pseudo, title=title, depth=depth + 1, max_items=max_items)
        return

    render_metric_card(title or "Value", _format_display(val))


def _phase_status(statuses: List[str]) -> tuple[str, str]:
    if all(s == "COMPLETED" for s in statuses):
        return "Complete", "ads-phase-done"
    if any(s == "ACTIVE" for s in statuses):
        return "In Progress", "ads-phase-active"
    if any(s == "COMPLETED" for s in statuses):
        return "Partial", "ads-phase-partial"
    return "Pending", "ads-phase-wait"


def render_compact_pipeline_stepper(
    all_steps: List[Dict[str, str]],
    get_status: Callable[[Dict[str, str]], str],
    render_step_detail: Callable[[Dict[str, str]], None],
) -> None:
    """Grouped vertical stepper — replaces HTML timeline and per-step expand buttons."""
    step_by_key = {step["key"]: step for step in all_steps}
    st.markdown('<div class="ads-stepper-compact">', unsafe_allow_html=True)

    for phase in _PIPELINE_PHASES:
        sub_steps = [step_by_key[key] for key in phase["steps"] if key in step_by_key]
        if not sub_steps:
            continue

        statuses = [get_status(step) for step in sub_steps]
        phase_label, phase_cls = _phase_status(statuses)
        done_count = sum(1 for status in statuses if status == "COMPLETED")
        expand = phase_label == "In Progress"

        with st.expander(
            f"{phase['icon']} {phase['label']} · {done_count}/{len(sub_steps)} · {phase_label}",
            expanded=expand,
        ):
            for step in sub_steps:
                status = get_status(step)
                icon = _STATUS_ICON.get(status, "○")
                st.markdown(
                    f"""
                    <div class="ads-step-row {phase_cls}">
                        <span class="ads-step-icon">{icon}</span>
                        <span class="ads-step-label">{step['label']}</span>
                        <span class="ads-step-badge">{status.title()}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if status in ("COMPLETED", "ACTIVE"):
                    with st.expander(f"{step['label']} details", expanded=False):
                        render_step_detail(step)

    st.markdown("</div>", unsafe_allow_html=True)


def render_ai_response_card(
    message: str,
    *,
    role: str = "assistant",
    timestamp: str = "",
) -> None:
    if role == "user":
        icon, card_cls, title = "🧑", "ads-ai-card-user", "You"
    else:
        icon, card_cls, title = "✨", "ads-ai-card-assistant", "AI Data Scientist"

    time_html = f'<span class="ads-ai-card-time">{timestamp}</span>' if timestamp else ""
    safe_message = message.replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="ads-ai-card {card_cls}">
            <div class="ads-ai-card-head">
                <span class="ads-ai-card-icon">{icon}</span>
                <span class="ads-ai-card-title">{title}</span>
                {time_html}
            </div>
            <div class="ads-ai-card-body">{safe_message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_slim_insight_panel(
    *,
    ai_status: str,
    current_task: str,
    progress_pct: int,
    recommendations: List[str],
) -> None:
    rec_html = "".join(f'<div class="ads-rec-item">{rec}</div>' for rec in recommendations[:3])
    if not rec_html:
        rec_html = '<div class="ads-rec-item">Run the pipeline for live recommendations.</div>'

    st.markdown(
        f"""
        <div class="ads-insight-panel ads-insight-slim">
            <div class="ads-insight-card">
                <h4>Pipeline</h4>
                <div class="ads-insight-stat"><span class="label">Status</span><span class="value">{ai_status}</span></div>
                <div class="ads-insight-stat"><span class="label">Stage</span><span class="value">{current_task}</span></div>
                <div class="ads-progress-bar"><div class="ads-progress-fill" style="width:{min(100, progress_pct)}%;"></div></div>
                <div class="ads-insight-stat"><span class="label">Progress</span><span class="value">{progress_pct}%</span></div>
            </div>
            <div class="ads-insight-card">
                <h4>Recommendations</h4>
                {rec_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
