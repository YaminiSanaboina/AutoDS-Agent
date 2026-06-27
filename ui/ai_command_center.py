"""
AI Command Center - Advanced AI Workflow Control Panel
Executes the MasterAutonomousPipeline in background with real-time result tracking and live UI updates.
Thread-safe state synchronization between background pipeline and Streamlit UI.
"""

import datetime
import json
import logging
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import matplotlib
from matplotlib.figure import Figure as MatplotlibFigure
try:
    from PIL import Image as PILImage
except Exception:
    PILImage = None

from agents.master_autonomous_pipeline import MasterAutonomousPipeline
from config import (
    ACCENT_COLOR,
    DANGER_COLOR,
    PRIMARY_COLOR,
    SECONDARY_COLOR,
    SMART_MODE_BUDGET_SECONDS,
    SUCCESS_COLOR,
    WARNING_COLOR,
)
from ui.components import (
    enterprise_panel,
    render_autonomous_run_button,
    render_ai_chat_workspace,
    render_command_bar,
    render_enterprise_sidebar,
    render_no_dataset_gate,
    section_header,
)
from ui.dashboard import (
    build_dashboard_context,
    render_home_page,
    render_reports_center,
    render_right_panel,
)
from ui.step_detail_panels import render_step_detail
from ui.saas_components import render_structure_as_cards
from utils.session_manager import (
    SessionKeys,
    get_autonomous_result,
    get_dataframe,
    get_dataset_name,
    get_problem_type,
    has_autonomous_result,
    has_dataset,
    reset_on_new_dataset,
    set_autonomous_result,
    set_dataframe,
)
from utils.health_score import compute_health_score
from utils.pipeline_bridge import apply_autonomous_result_to_session
from utils.thread_safe_execution_state import ThreadSafeExecutionState


# ==============================================================================
# GLOBAL EXECUTION STATE (THREAD-SAFE)
# ==============================================================================

# Single global instance for thread-safe communication
_execution_state = ThreadSafeExecutionState()


# ==============================================================================
# LOGGING UTILITIES
# ==============================================================================

_logger = logging.getLogger(__name__)


def _log_info(message: str) -> None:
    _logger.debug("[AI Command Center] %s", message)


def _log_error(message: str) -> None:
    _logger.warning("[AI Command Center ERROR] %s", message)


def _log_callback(stage: str, status: str, percent: int) -> None:
    _logger.debug("[AI Command Center] Callback: stage=%s status=%s progress=%s%%", stage, status, percent)


# ==============================================================================
# WORKFLOW STAGE DEFINITIONS
# ==============================================================================

def _workflow_steps() -> List[Dict[str, str]]:
    """Define all 15 stages of the autonomous ML workflow."""
    return [
        {"key": "dataset_upload", "label": "Dataset Upload", "output_key": "dataset_name", "subtitle": "Add a CSV and verify the dataset schema."},
        {"key": "dataset_intelligence", "label": "Data Profiling", "output_key": "dataset_report", "subtitle": "AI analyzes features, distributions, and target relationships."},
        {"key": "data_cleaning", "label": "Data Cleaning & Preprocessing", "output_key": "cleaning_results", "subtitle": "Automated cleaning recommendations and quality improvements."},
        {"key": "eda", "label": "Exploratory Data Analysis (EDA)", "output_key": "eda_results", "subtitle": "Visualize distributions, correlations, and anomalies."},
        {"key": "feature_engineering", "label": "Feature Engineering", "output_key": "feature_engineering_results", "subtitle": "Generate candidate transformations and feature suggestions."},
        {"key": "automl", "label": "Model Training", "output_key": "model_results", "subtitle": "Train candidate models and select the best performer."},
        {"key": "model_comparison", "label": "Model Comparison", "output_key": "model_comparison", "subtitle": "Compare top models and pick the best strategy."},
        {"key": "explainability", "label": "Explainability (Feature Importance / SHAP)", "output_key": "explainability_results", "subtitle": "Understand model drivers and feature impact."},
        {"key": "ai_ethics_trust", "label": "Trust Score Assessment", "output_key": "ai_trust_results", "subtitle": "Validate model fairness, safety, and trust."},
        {"key": "self_improvement", "label": "Model Evaluation", "output_key": "improvement_history", "subtitle": "Refine model pipelines based on performance feedback."},
        {"key": "deployment_readiness", "label": "Deployment Readiness Assessment", "output_key": "deployment_readiness", "subtitle": "Assess risk and readiness for production deployment."},
        {"key": "monitoring", "label": "Production Monitoring", "output_key": "monitoring_results", "subtitle": "Track drift, retraining, and production health."},
        {"key": "ai_decision", "label": "Final AI Decision", "output_key": "recommendation", "subtitle": "Receive the AI command center recommendation."},
        {"key": "prediction", "label": "Prediction Engine", "output_key": "prediction_results", "subtitle": "Run sample predictions with the selected model."},
        {"key": "pdf_report", "label": "Executive Report Generation", "output_key": "final_report", "subtitle": "Review and export the executive AI report."},
    ]



# ==============================================================================
# THREAD-SAFE STATE SYNCHRONIZATION
# ==============================================================================

def _safe_set_session_key(key: str, value: Any) -> bool:
    """
    Set session_state[key] = value only if the value changed.
    Returns True if the value was changed, False if it was the same.
    This prevents spurious reruns caused by setting the same value repeatedly.
    """
    current = st.session_state.get(key)
    if current != value:
        st.session_state[key] = value
        return True
    return False


def _sync_execution_state_to_session() -> bool:
    """
    Poll execution_state and sync updates to st.session_state.
    Called on every rerun to pick up background thread updates.

    Returns True when fresh pipeline results were applied and the UI should rerun once.
    """
    global _execution_state

    from utils.session_manager import get_autonomous_result, has_autonomous_result, has_dataset

    needs_ui_refresh = False

    thread_status = _execution_state.get_status()

    if not has_dataset():
        if thread_status == "running":
            return
        _execution_state.reset()
        _safe_set_session_key(SessionKeys.PIPELINE_RUNNING, False)
        return

    status = thread_status
    if status == "running":
        _safe_set_session_key(SessionKeys.PIPELINE_RUNNING, True)
    elif status in ("completed", "error", "idle"):
        _safe_set_session_key(SessionKeys.PIPELINE_RUNNING, False)
    
    # Sync progress — only write if changed
    progress = _execution_state.get_progress()
    _safe_set_session_key(SessionKeys.PIPELINE_PROGRESS, progress)
    
    # Sync current stage — only write if changed
    current_stage = _execution_state.get_current_stage()
    if current_stage:
        _safe_set_session_key(SessionKeys.PIPELINE_CURRENT_STAGE, current_stage)
    
    # Sync elapsed time — only write if changed
    elapsed = _execution_state.get_elapsed_time()
    _safe_set_session_key(SessionKeys.PIPELINE_ELAPSED_TIME, elapsed)
    
    # Sync stage results — never wipe persisted session results when execution state is idle
    all_results = _execution_state.get_all_stage_results()
    if all_results:
        _safe_set_session_key(SessionKeys.PIPELINE_STAGE_RESULTS, all_results)
    elif not st.session_state.get(SessionKeys.PIPELINE_STAGE_RESULTS):
        from utils.pipeline_bridge import build_stage_results_from_output, build_stage_statuses_from_results
        from utils.session_manager import get_autonomous_result, has_autonomous_result, has_dataset

        cached = get_autonomous_result()
        if (
            has_dataset()
            and thread_status != "running"
            and has_autonomous_result()
            and isinstance(cached, dict)
        ):
            hydrated = build_stage_results_from_output(cached)
            if hydrated:
                _safe_set_session_key(SessionKeys.PIPELINE_STAGE_RESULTS, hydrated)
                _safe_set_session_key(SessionKeys.PIPELINE_STAGE_STATUSES, build_stage_statuses_from_results(hydrated))
                _safe_set_session_key(SessionKeys.PIPELINE_COMPLETED_STAGES, list(hydrated.keys()))
    
    # Sync stage statuses (for computing timeline card states) — only write if changed
    all_statuses = _execution_state.get_all_stage_statuses()
    if _safe_set_session_key(SessionKeys.PIPELINE_STAGE_STATUSES, all_statuses):
        if all_statuses:
            completed_stages = [s for s, status in all_statuses.items() if status == "completed"]
            _safe_set_session_key(SessionKeys.PIPELINE_COMPLETED_STAGES, completed_stages)
    
    # Sync events — only write if changed
    events = _execution_state.get_events()
    if events:
        _safe_set_session_key(SessionKeys.PIPELINE_EVENTS, events[-20:])
    
    # Sync final result once per execution id
    final_result = _execution_state.get_final_result()
    if final_result:
        applied_id = st.session_state.get("_applied_pipeline_result_id")
        state_id = _execution_state.get_execution_state_id() or st.session_state.get(SessionKeys.PIPELINE_EXECUTION_STATE_ID)
        if state_id and applied_id != state_id:
            apply_autonomous_result_to_session(final_result)
            st.session_state["_applied_pipeline_result_id"] = state_id
            st.session_state["_pipeline_just_finished"] = True
            st.session_state["report_context_version"] = state_id
            needs_ui_refresh = True
            _execution_state.clear_final_result()
    
    # Sync error — only write if changed
    error = _execution_state.get_error()
    _safe_set_session_key(SessionKeys.PIPELINE_ERROR, error)

    # Sync execution state ID — only write if changed
    state_id = _execution_state.get_execution_state_id()
    if state_id:
        _safe_set_session_key(SessionKeys.PIPELINE_EXECUTION_STATE_ID, state_id)

    # Sync last update time to allow controlled rerun polling — only write if changed
    last_update_time = _execution_state.get_last_update_time()
    if last_update_time:
        _safe_set_session_key(SessionKeys.PIPELINE_LAST_SYNC_TIME, last_update_time.isoformat())

    return needs_ui_refresh


def _run_pipeline_background(
    dataset: pd.DataFrame,
    dataset_name: str,
    project_goal: str,
    constraints: Dict[str, Any],
    execution_state_id: str,
) -> None:
    """
    Run the MasterAutonomousPipeline in background thread.
    Updates execution_state (thread-safe) with progress and results.
    Main render loop polls execution_state and syncs to st.session_state.
    """
    global _execution_state

    _execution_state.set_execution_state_id(execution_state_id)
    _log_info(f"Pipeline execution starting (state_id={execution_state_id})...")
    _execution_state.set_status("running")
    _execution_state.set_start_time(datetime.datetime.utcnow())

    try:
        _log_info(f"Dataset: {dataset_name}, Goal: {project_goal}")

        # Initialize pipeline
        pipeline = MasterAutonomousPipeline()

        # Progress callback - called by pipeline at each stage
        def progress_callback(event: Dict[str, Any]) -> None:
            """Update execution_state with pipeline progress (thread-safe)."""
            stage = event.get("stage", "unknown")
            status = event.get("status", "running")
            percent = event.get("percent", 0)
            result = event.get("result")

            _log_callback(stage, status, percent)

            # Update state (all thread-safe)
            _execution_state.set_progress(percent)
            _execution_state.set_current_stage(stage)

            # Track stage status
            if status == "started":
                _execution_state.set_stage_status(stage, "active")
            elif status == "completed":
                _execution_state.set_stage_status(stage, "completed")

            # Store result if provided
            if result is not None:
                _execution_state.add_stage_result(stage, result)
                _log_info(f"Stored result for stage {stage}: {type(result).__name__}")

            # Add event to history
            _execution_state.add_event({
                "stage": stage,
                "status": status,
                "percent": percent,
            })

            # Update elapsed time
            start_time = _execution_state.get_start_time()
            if start_time:
                elapsed = (datetime.datetime.utcnow() - start_time).total_seconds()
                _execution_state.set_elapsed_time(elapsed)

        _log_info("Starting pipeline execution...")

        # Execute pipeline
        output = pipeline.run_pipeline(
            dataset=dataset,
            dataset_name=dataset_name,
            project_goal=project_goal,
            constraints=constraints,
            smart_mode=True,
            max_seconds=SMART_MODE_BUDGET_SECONDS,
            progress_callback=progress_callback,
            use_cache=True,
        )

        _log_info("Pipeline execution completed successfully")

        # Store final result and update state
        _execution_state.set_final_result(output)
        _execution_state.set_status("completed")
        _execution_state.set_progress(100)

        # Final elapsed time and persist to session for all pages
        start_time = _execution_state.get_start_time()
        if start_time:
            elapsed = (datetime.datetime.utcnow() - start_time).total_seconds()
            _execution_state.set_elapsed_time(elapsed)

    except Exception as exc:
        _log_error(f"Pipeline execution failed: {exc}")
        _log_error(traceback.format_exc())
        # Record final elapsed time even on error
        start_time = _execution_state.get_start_time()
        if start_time:
            elapsed = (datetime.datetime.utcnow() - start_time).total_seconds()
            _execution_state.set_elapsed_time(elapsed)
        try:
            from agents.self_healing_agent import SelfHealingAgent

            healer = SelfHealingAgent()
            analysis = healer.analyze_error(str(exc))
            fix = healer.recommend_fix(analysis)
            friendly = fix.get("explanation") or analysis.get("root_cause") or str(exc)
            action = fix.get("recommended_action")
            if action:
                friendly = f"{friendly} Recommended action: {action}"
            _execution_state.set_error(friendly)
        except Exception:
            _execution_state.set_error(str(exc))
        _execution_state.set_status("error")


# ==============================================================================
# UI HELPER FUNCTIONS
# ==============================================================================

def _status_color(status: str) -> str:
    """Get color for status badge."""
    if status == "COMPLETED":
        return SUCCESS_COLOR
    elif status == "ACTIVE":
        return ACCENT_COLOR
    else:  # WAITING
        return WARNING_COLOR


def _get_stage_status(step: Dict[str, str]) -> str:
    """Get current status for a workflow step."""
    stage_statuses = st.session_state.get(SessionKeys.PIPELINE_STAGE_STATUSES, {})
    backend_key = _get_stage_result_key(step["key"]) or step["key"]
    backend_status = stage_statuses.get(backend_key)
    if backend_status == "completed":
        return "COMPLETED"
    if backend_status == "active":
        return "ACTIVE"

    completed = st.session_state.get(SessionKeys.PIPELINE_COMPLETED_STAGES, [])
    current = st.session_state.get(SessionKeys.PIPELINE_CURRENT_STAGE)

    if step["key"] in completed or backend_key in completed:
        return "COMPLETED"
    if step["key"] == current or backend_key == current:
        if st.session_state.get(SessionKeys.PIPELINE_RUNNING):
            return "ACTIVE"
    return "WAITING"


def _open_panel(title: str, subtitle: Optional[str] = None) -> None:
    """Render panel heading (legacy helper — use enterprise_panel for new code)."""
    sub = f'<p class="ads-panel-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="ads-panel-heading"><div class="card-title">{title}</div>{sub}</div>',
        unsafe_allow_html=True,
    )


def _close_panel() -> None:
    """Legacy no-op — split HTML wrappers caused raw </div> text in Streamlit."""
    return


_STAGE_RESULT_KEY_BY_UI_KEY = {
    "dataset_upload": "dataset_upload",
    "dataset_intelligence": "dataset_intelligence",
    "data_cleaning": "data_cleaning",
    "eda": "eda",
    "feature_engineering": "feature_engineering",
    "automl": "automl",
    "model_comparison": "model_comparison",
    "explainability": "explainability",
    "ai_ethics_trust": "ai_ethics_trust",
    "self_improvement": "self_improvement",
    "deployment_readiness": "deployment_readiness",
    "monitoring": "monitoring",
    "ai_decision": "ai_decision",
    "prediction": "prediction",
    "pdf_report": "pdf_report",
    "hyperparameter_optimization": "hyperparameter_optimization",
}


def _get_stage_result_key(ui_step_key: str) -> Optional[str]:
    """Map a UI workflow step key to the backend stage result key."""
    return _STAGE_RESULT_KEY_BY_UI_KEY.get(ui_step_key)


def _render_stage_output(result: Any, title: str = "Stage result") -> None:
    """Render a generic stage output object (recursive) based on its type.

    Supports: pandas.DataFrame, dict, list, matplotlib figures, plotly figures,
    PIL images, bytes (images), and primitives. Any rendering error for a
    component is caught and shown as a warning so the page does not crash.
    """
    def _inner(val: Any, ctx_title: Optional[str] = None):
        try:
            if val is None:
                st.info("No results available for this component.")
                return

            # DataFrame
            if isinstance(val, pd.DataFrame):
                if val.empty:
                    st.info("Empty table.")
                else:
                    st.dataframe(val, width="stretch")
                return

            # Dict / list: professional card layout
            if isinstance(val, (dict, list)):
                render_structure_as_cards(val, title=ctx_title)
                return

            # Matplotlib figure
            if MatplotlibFigure and isinstance(val, MatplotlibFigure):
                try:
                    st.pyplot(val)
                except Exception as e:
                    st.warning(f"Unable to render matplotlib figure: {e}")
                return

            # Plotly figure detection
            try:
                if "plotly" in str(type(val)).lower():
                    st.plotly_chart(val, width="stretch")
                    return
            except Exception:
                pass

            # PIL Image
            if PILImage and isinstance(val, PILImage.Image):
                try:
                    st.image(val)
                except Exception as e:
                    st.warning(f"Unable to render image: {e}")
                return

            # Bytes - try to interpret as image
            if isinstance(val, (bytes, bytearray)):
                try:
                    import io
                    from PIL import Image

                    img = Image.open(io.BytesIO(val))
                    st.image(img)
                    return
                except Exception:
                    # Not an image; fall through to write
                    pass

            # Fallback for numpy arrays and other primitives
            st.write(val)
        except Exception as exc:
            st.warning(f"Unable to render component: {exc}")

    # Start rendering
    if title:
        try:
            st.markdown(f"### {title}")
        except Exception:
            pass
    _inner(result, title)


def _get_stage_output_key(ui_step_key: str) -> Optional[str]:
    """Resolve the backend stage result key for a UI workflow step."""
    return _STAGE_RESULT_KEY_BY_UI_KEY.get(ui_step_key, ui_step_key)


def _get_stage_result(stage_results: Dict[str, Any], ui_step_key: str) -> Any:
    """Get the stage result for a given UI workflow step."""
    stage_key = _get_stage_output_key(ui_step_key)
    return stage_results.get(stage_key)


def _launch_autonomous_pipeline() -> None:
    """Start the MasterAutonomousPipeline in a background thread (existing execution path)."""
    global _execution_state

    if not has_dataset():
        return

    _log_info("Start button clicked, launching background thread")
    dataset = get_dataframe()
    dataset_name = get_dataset_name()
    project_goal = st.session_state.get("project_goal", "Predict the target variable")
    constraints = {"notes": st.session_state.get("ai_command_constraints", "")}

    if has_autonomous_result():
        cached_output = get_autonomous_result()
        if isinstance(cached_output, dict):
            cached_dataset = cached_output.get("dataset_name")
            cached_goal = cached_output.get("project_goal")
            cached_constraints = cached_output.get("constraints")
            if cached_dataset == dataset_name and cached_goal == project_goal and cached_constraints == constraints:
                apply_autonomous_result_to_session(cached_output)
                st.session_state[SessionKeys.PIPELINE_RUNNING] = False
                st.session_state[SessionKeys.PIPELINE_PROGRESS] = 100
                st.session_state[SessionKeys.PIPELINE_CURRENT_STAGE] = "pdf_report"
                st.session_state["_pipeline_just_finished"] = True
                st.rerun()
                return

    _execution_state.reset()
    execution_state_id = uuid.uuid4().hex
    st.session_state.pop("_applied_pipeline_result_id", None)
    st.session_state[SessionKeys.PIPELINE_RUNNING] = True
    st.session_state[SessionKeys.PIPELINE_PROGRESS] = 0
    st.session_state[SessionKeys.PIPELINE_CURRENT_STAGE] = "initializing"
    st.session_state[SessionKeys.PIPELINE_COMPLETED_STAGES] = []
    st.session_state[SessionKeys.PIPELINE_STAGE_STATUSES] = {}
    st.session_state[SessionKeys.PIPELINE_STAGE_RESULTS] = {}
    st.session_state[SessionKeys.PIPELINE_ERROR] = None
    st.session_state[SessionKeys.PIPELINE_EXECUTION_STATE_ID] = execution_state_id
    st.session_state[SessionKeys.PIPELINE_EVENTS] = []
    st.session_state[SessionKeys.PIPELINE_START_TIME] = datetime.datetime.utcnow()
    st.session_state[SessionKeys.PIPELINE_ELAPSED_TIME] = 0  # Reset timer
    st.session_state[SessionKeys.AUTONOMOUS_RESULT] = None
    thread = threading.Thread(
        target=_run_pipeline_background,
        args=(dataset, dataset_name, project_goal, constraints, execution_state_id),
        daemon=True,
    )
    thread.start()
    st.rerun()


def _pipeline_run_clicked(*, key: str) -> bool:
    """Render the run button and launch the pipeline when clicked."""
    is_running = st.session_state.get(SessionKeys.PIPELINE_RUNNING, False)
    disabled = not has_dataset() or is_running
    if render_autonomous_run_button(key=key, disabled=disabled, is_running=is_running):
        _launch_autonomous_pipeline()
        return True
    return False


def _render_stat_grid(items: List[Dict[str, Any]], columns: int = 2) -> None:
    """Render a grid of stat cards."""
    cards = []
    for item in items:
        hint = item.get("hint")
        card_html = (
            "<div class='card-panel-small' style='min-height:118px;padding:1rem;'>"
            f"<div style='color:#64748B;font-size:0.85rem;font-weight:700;margin-bottom:0.55rem;'>{item['label']}</div>"
            f"<div style='font-size:1.4rem;font-weight:800;color:#0F172A;'>{item['value']}</div>"
            + (f"<div style='margin-top:0.45rem;color:#64748B;font-size:0.88rem;'>{hint}</div>" if hint else "")
            + "</div>"
        )
        cards.append(card_html)

    st.markdown(
        f"<div style='display:grid;grid-template-columns:repeat({columns},minmax(0,1fr));gap:16px;margin-top:16px;'>" + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def _dataset_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Get basic profile stats from dataset."""
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_pct": float(df.isnull().mean().mean() * 100),
        "duplicates": int(df.duplicated().sum()),
        "problem_type": get_problem_type(),
    }


# ==============================================================================
# TIMELINE RENDERING
# ==============================================================================

def _render_timeline(output: Optional[Dict[str, Any]] = None) -> None:
    """Compact vertical stepper grouped by pipeline phase."""
    from ui.saas_components import render_compact_pipeline_stepper

    steps = _workflow_steps()
    stage_results = st.session_state.get(SessionKeys.PIPELINE_STAGE_RESULTS, {})

    def _render_step(step: Dict[str, str]) -> None:
        key = step["key"]
        backend_key = _get_stage_output_key(key) or key
        render_step_detail(
            key,
            output,
            stage_results,
            backend_key=backend_key,
            open_panel=_open_panel,
            close_panel=_close_panel,
            render_stat_grid=_render_stat_grid,
            render_stage_output=_render_stage_output,
        )

    render_compact_pipeline_stepper(steps, _get_stage_status, _render_step)


# ==============================================================================
# STEP DETAILS RENDERING (ACTUAL RESULTS FROM PIPELINE)
# ==============================================================================

def _render_step_details(output: Optional[Dict[str, Any]] = None) -> None:
    """Render detailed results for expanded steps (fallback for stages opened elsewhere)."""
    stage_results = st.session_state.get(SessionKeys.PIPELINE_STAGE_RESULTS, {})
    for item in _workflow_steps():
        key = item["key"]
        if not st.session_state.get(f"open_step_{key}"):
            continue
        backend_key = _get_stage_output_key(key) or key
        render_step_detail(
            key,
            output,
            stage_results,
            backend_key=backend_key,
            open_panel=_open_panel,
            close_panel=_close_panel,
            render_stat_grid=_render_stat_grid,
            render_stage_output=_render_stage_output,
        )


def _render_hero_dashboard(output: Optional[Dict[str, Any]] = None) -> None:
    """Premium home dashboard hero with animated metric cards and run control."""
    ctx = build_dashboard_context(output)
    subtitle = (
        "Autonomous Data Science Platform — I analyzed your dataset and created a complete AI strategy."
        if output
        else "Autonomous Data Science Platform — upload a dataset and run the autonomous AI scientist."
    )
    hero_col, run_col = st.columns([3.4, 1.2])
    with hero_col:
        render_compact_hero(
            title="AI Command Center",
            subtitle=subtitle,
            health=ctx["health_display"],
            confidence=ctx["confidence_display"],
            best_model=str(ctx["best_model"]),
            deployment=ctx["deployment_label"],
        )
    with run_col:
        if has_dataset():
            _pipeline_run_clicked(key="hero_run_autods")


_SAMPLE_DATASETS = {
    "Iris (Classification)": Path("data/Iris.csv"),
    "Titanic (Classification)": Path("data/Titanic-Dataset.csv"),
    "Heart Disease (Classification)": Path("data/Heart-Disease.csv"),
    "Housing (Regression)": Path("data/Housing.csv"),
    "Wine Quality (Regression)": Path("data/Wine-Quality.csv"),
}


_ANALYSIS_STAGE_STATUS = {
    "initializing": "Stage 0: Initializing analysis engine",
    "dataset_upload": "Stage 1: Dataset Profiling",
    "dataset_intelligence": "Stage 1: Dataset Profiling",
    "data_cleaning": "Stage 2: Data Quality Assessment",
    "eda": "Stage 3: EDA Generation",
    "feature_engineering": "Stage 4: Feature Engineering",
    "automl": "Stage 5: AutoML Training",
    "model_comparison": "Stage 6: Model Evaluation",
    "explainability": "Stage 7: SHAP Analysis",
    "ai_ethics_trust": "Stage 8: Trust Analysis",
    "self_improvement": "Stage 8: Trust Analysis",
    "deployment_readiness": "Stage 9: Recommendations",
    "monitoring": "Stage 9: Recommendations",
    "ai_decision": "Stage 9: Recommendations",
    "prediction": "Stage 9: Recommendations",
    "pdf_report": "Stage 10: Report Generation",
}


def _render_home_upload() -> None:
    """Compact dataset upload and configuration on the Home page."""
    with enterprise_panel("Upload Dataset", "Load a CSV or sample dataset to begin autonomous analysis."):
        uploaded_file = st.file_uploader(
            "CSV dataset",
            type=["csv"],
            key="home_dataset_upload",
            help="Drag a file here or click to browse.",
        )
        sample_col, load_col = st.columns([2, 1])
        with sample_col:
            sample_choice = st.selectbox(
                "Sample dataset",
                ["— Select sample —"] + list(_SAMPLE_DATASETS.keys()),
                key="home_sample_dataset",
            )
        with load_col:
            st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
            if st.button("Load Sample", key="home_load_sample", width="stretch"):
                if sample_choice.startswith("—"):
                    st.warning("Choose a sample dataset first.")
                else:
                    sample_path = _SAMPLE_DATASETS[sample_choice]
                    if sample_path.exists():
                        df = pd.read_csv(sample_path)
                        reset_on_new_dataset(df, sample_path.name)
                        st.success(f"Loaded {sample_path.name}")
                        st.rerun()
                    else:
                        st.error(f"Sample file not found: {sample_path}")

        if uploaded_file is not None:
            try:
                uploaded_name = uploaded_file.name
                processed_name = st.session_state.get("_home_dataset_upload_processed")
                if processed_name != uploaded_name or not has_dataset():
                    df = pd.read_csv(uploaded_file)
                    reset_on_new_dataset(df, uploaded_name)
                    st.session_state["_home_dataset_upload_processed"] = uploaded_name
                    st.success("Dataset loaded successfully")
                    st.rerun()
            except Exception as exc:
                st.error(f"Unable to parse dataset: {exc}")

        if has_dataset():
            st.text_input(
                "Project goal",
                value=st.session_state.get("project_goal", "Predict the target variable"),
                key="project_goal",
            )
            st.text_area(
                "Constraints or notes",
                value=st.session_state.get("ai_command_constraints", ""),
                key="ai_command_constraints",
                height=80,
            )


def _render_pipeline_controls(output: Optional[Dict[str, Any]]) -> None:
    """Run Autonomous Analysis — single action to execute the full pipeline."""

    if not has_dataset():
        st.warning("Upload a dataset above to enable autonomous analysis.")
        return

    is_running = st.session_state.get(SessionKeys.PIPELINE_RUNNING, False)
    disabled = is_running
    if st.button(
        "Run Autonomous Analysis",
        key="home_run_autonomous_analysis",
        type="primary",
        width="stretch",
        disabled=disabled,
    ):
        _launch_autonomous_pipeline()

    if is_running:
        return

    if has_autonomous_result():
        elapsed = float(st.session_state.get(SessionKeys.PIPELINE_ELAPSED_TIME, 0) or 0)
        elapsed_text = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        st.success(f"Analysis complete in {elapsed_text}. Open **Reports** for the full executive report.")

    error = st.session_state.get(SessionKeys.PIPELINE_ERROR)
    if error:
        st.error(f"Pipeline error: {error}")
    recovery_messages = st.session_state.get("pipeline_recovery_messages") or []
    if recovery_messages:
        with st.expander("Recovery guidance", expanded=False):
            for message in recovery_messages[:5]:
                st.warning(message)


def _render_workspace_view(view: str, output: Optional[Dict[str, Any]]) -> None:
    """Route center workspace — Home or Reports."""
    from ui.components import normalize_nav_view

    view = normalize_nav_view(view)

    if view == "home":
        render_home_page(
            output,
            upload_fn=_render_home_upload,
            run_fn=lambda: _render_pipeline_controls(output),
        )
    elif view == "reports":
        render_reports_center(output)
    else:
        st.session_state["enterprise_nav_view"] = "home"
        st.rerun()


def _render_ai_chat_assistant() -> None:
    """ChatGPT-style assistant wired to the shared legacy assistant backend."""
    from ui.legacy_pages.ai_assistant import generate_response, SUGGESTED_QUESTIONS

    render_ai_chat_workspace(
        "ai_command_chat",
        generate_response,
        title="AI Data Scientist Assistant",
        subtitle="Ask about features, models, charts, deployment readiness, and reports.",
        suggested_questions=SUGGESTED_QUESTIONS,
        chat_input_placeholder="Ask: What are the important features? Which model is best?",
        open_panel_fn=_open_panel,
        close_panel_fn=_close_panel,
    )


def _render_image_analysis_workspace() -> None:
    """Render the standalone image classification demo."""
    from ui.image_analysis_demo import render_image_analysis_demo

    render_image_analysis_demo()


# ==============================================================================
# MAIN RENDER FUNCTION
# ==============================================================================

def render():
    """Main AI Command Center render function with enterprise layout."""
    global _execution_state

    needs_ui_refresh = _sync_execution_state_to_session()

    output = get_autonomous_result() if has_autonomous_result() else None
    ctx = build_dashboard_context(output)

    from ui.components import normalize_nav_view

    st.session_state.setdefault("enterprise_nav_view", "home")
    active_view = normalize_nav_view(st.session_state["enterprise_nav_view"])

    if st.session_state.pop("_pipeline_just_finished", False) and has_autonomous_result():
        st.session_state["_show_analysis_complete_banner"] = True

    st.session_state["enterprise_nav_view"] = active_view

    render_command_bar(
        version="Enterprise v2.0",
        production_badge=ctx["production_badge"],
        production_class=ctx["production_class"],
        smart_mode=True,
        timer_text=ctx["timer_text"],
        session_id=str(ctx["session_id"]),
        pipeline_status=ctx["pipeline_status"],
        dataset_status=ctx["dataset_status"],
        current_stage=ctx["current_stage"],
    )

    if active_view == "reports":
        left_col, center_col = st.columns([1, 4.2], gap="medium")
    else:
        left_col, center_col = st.columns([1, 4.2], gap="medium")

    with left_col:
        selected = render_enterprise_sidebar(
            active_view=active_view,
            dataset_status=ctx["dataset_status"],
            health_score=ctx["health_display"],
            current_stage=ctx["current_stage"],
        )
        selected = normalize_nav_view(selected)
        if selected != active_view:
            st.session_state["enterprise_nav_view"] = selected
            st.rerun()

    with center_col:
        _render_workspace_view(active_view, output)

    is_running = st.session_state.get(SessionKeys.PIPELINE_RUNNING, False)
    if is_running:
        last_update_time = _execution_state.get_last_update_time()
        last_sync = st.session_state.get(SessionKeys.PIPELINE_LAST_SYNC_TIME)
        if last_update_time is not None and last_update_time.isoformat() != last_sync:
            st.session_state[SessionKeys.PIPELINE_LAST_SYNC_TIME] = last_update_time.isoformat()
            time.sleep(0.3)
            st.rerun()
    elif needs_ui_refresh:
        time.sleep(0.15)
        st.rerun()
