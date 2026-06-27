import streamlit as st
import pandas as pd
import numpy as np

from utils.safe_checks import coalesce_dict
class SessionKeys:
    CURRENT_PAGE = 'current_page'
    UPLOADED_DF = 'uploaded_dataframe'
    DF = 'df'
    DF_ORIGINAL = 'df_original'
    DATASET_NAME = 'dataset_name'
    UPLOAD_FILENAME = 'upload_filename'
    DATASET_METADATA = 'dataset_metadata'
    DATASET_LOADED = 'dataset_loaded'
    CLEANING_REPORT = 'cleaning_report'
    CLEANING_ISSUES = 'cleaning_issues'
    CLEANING_HISTORY = 'cleaning_history'
    HEALTH_BEFORE = 'health_before_cleaning'
    HEALTH_AFTER = 'health_after_cleaning'
    EDA_GENERATED = 'eda_generated'
    EDA_SUMMARY = 'eda_summary'
    EDA_NUMERICAL_COLUMNS = 'eda_numerical_columns'
    EDA_CATEGORICAL_COLUMNS = 'eda_categorical_columns'
    EDA_SELECTED_FEATURE = 'eda_selected_feature'
    EDA_SELECTED_FEATURE_2 = 'eda_selected_feature_2'
    EDA_CHART_TYPE = 'eda_chart_type'
    EDA_ACTIVE_TAB = 'eda_active_tab'
    EDA_INSIGHTS = 'eda_insights'
    MODEL_TRAINED = 'model_trained'
    TARGET_COLUMN = 'target_column'
    SELECTED_ALGORITHMS = 'selected_algorithms'
    PROBLEM_TYPE = 'problem_type'
    X_DATA = 'X_data'
    Y_DATA = 'y_data'
    AUTONOMOUS_RESULT = 'autonomous_result'
    TARGET_ENCODER = 'target_encoder'
    BEST_MODEL = 'best_model'
    BEST_MODEL_NAME = 'best_model_name'
    RESULTS = 'results'
    MODEL_LEADERBOARD = 'model_leaderboard'
    TRAINED_MODELS = 'trained_models'
    MODEL_METRICS = 'model_metrics'
    CONFUSION_MATRIX = 'confusion_matrix'
    ROC_DATA = 'roc_data'
    FEATURE_IMPORTANCE = 'feature_importance'
    SHAP_COMPUTED = 'shap_computed'
    SHAP_VALUES = 'shap_values'
    SHAP_IMPORTANCE = 'shap_importance'
    SHAP_POSITIVE_NEGATIVE = 'shap_positive_negative'
    AI_INSIGHTS = 'ai_insights'
    CONFIDENCE_SCORE = 'confidence_score'
    RECOMMENDATIONS = 'recommendations'
    REPORT_GENERATED = 'report_generated'
    REPORT_PATH = 'report_path'
    REPORT_PAYLOAD = 'report_payload'
    # Pipeline execution tracking
    PIPELINE_RUNNING = 'pipeline_running'
    PIPELINE_PROGRESS = 'pipeline_progress'
    PIPELINE_CURRENT_STAGE = 'pipeline_current_stage'
    PIPELINE_COMPLETED_STAGES = 'pipeline_completed_stages'
    PIPELINE_STAGE_STATUSES = 'pipeline_stage_statuses'
    PIPELINE_STAGE_RESULTS = 'pipeline_stage_results'
    PIPELINE_ERROR = 'pipeline_error'
    PIPELINE_EXECUTION_STATE_ID = 'execution_state_id'
    PIPELINE_LAST_SYNC_TIME = 'pipeline_last_sync_time'
    PIPELINE_START_TIME = 'pipeline_start_time'
    PIPELINE_ELAPSED_TIME = 'pipeline_elapsed_time'
    PIPELINE_EVENTS = 'pipeline_events'
    PIPELINE_EXECUTED = 'pipeline_executed'
    VALIDATION_REPORT = 'validation_report'
    VALIDATION_SCORE = 'validation_score'
    BEST_SCORE = 'best_score'
    ANALYSIS_PROFILE = 'analysis_profile'
    # Executive metrics (single source of truth)
    EXECUTIVE_METRICS = 'executive_metrics'
    PIPELINE_COMPLETE = 'pipeline_complete'

def init_session():
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'ai_command_center'
    _sync_df_aliases()
    # Initialize common session keys with safe defaults to avoid runtime KeyErrors
    defaults = {
        SessionKeys.PROBLEM_TYPE: 'Classification',
        SessionKeys.RESULTS: {},
        SessionKeys.AUTONOMOUS_RESULT: None,
        SessionKeys.REPORT_GENERATED: False,
        SessionKeys.SHAP_COMPUTED: False,
        'prediction_history': [],
        SessionKeys.BEST_MODEL_NAME: None,
        SessionKeys.BEST_MODEL: None,
        SessionKeys.CONFIDENCE_SCORE: None,
        SessionKeys.EDA_GENERATED: False,
        # Pipeline tracking
        SessionKeys.PIPELINE_RUNNING: False,
        SessionKeys.PIPELINE_PROGRESS: 0,
        SessionKeys.PIPELINE_CURRENT_STAGE: None,
        SessionKeys.PIPELINE_COMPLETED_STAGES: [],
        SessionKeys.PIPELINE_STAGE_RESULTS: {},
        SessionKeys.PIPELINE_STAGE_STATUSES: {},
        SessionKeys.PIPELINE_ERROR: None,
        SessionKeys.PIPELINE_EXECUTION_STATE_ID: None,
        SessionKeys.PIPELINE_LAST_SYNC_TIME: None,
        SessionKeys.PIPELINE_START_TIME: None,
        SessionKeys.PIPELINE_ELAPSED_TIME: 0,
        SessionKeys.PIPELINE_EVENTS: [],
        SessionKeys.PIPELINE_EXECUTED: False,
        SessionKeys.VALIDATION_REPORT: None,
        SessionKeys.VALIDATION_SCORE: None,
        # Executive metrics (single source of truth for all UI pages)
        SessionKeys.EXECUTIVE_METRICS: {},
        SessionKeys.PIPELINE_COMPLETE: False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not has_dataset():
        st.session_state[SessionKeys.AUTONOMOUS_RESULT] = None
        st.session_state[SessionKeys.PIPELINE_RUNNING] = False
        st.session_state[SessionKeys.PIPELINE_PROGRESS] = 0
        st.session_state[SessionKeys.PIPELINE_CURRENT_STAGE] = None
        st.session_state[SessionKeys.PIPELINE_STAGE_STATUSES] = {}
        st.session_state[SessionKeys.PIPELINE_STAGE_RESULTS] = {}
        st.session_state[SessionKeys.PIPELINE_COMPLETED_STAGES] = []
        st.session_state[SessionKeys.PIPELINE_EXECUTED] = False
        st.session_state[SessionKeys.REPORT_GENERATED] = False
        st.session_state[SessionKeys.EXECUTIVE_METRICS] = {}
        st.session_state[SessionKeys.PIPELINE_COMPLETE] = False
        st.session_state.pop("_applied_pipeline_result_id", None)
        st.session_state.pop("pipeline_validation_warnings", None)

    if has_dataset() and not st.session_state.get(SessionKeys.PIPELINE_RUNNING) and st.session_state.get(SessionKeys.AUTONOMOUS_RESULT) and not st.session_state.get(SessionKeys.PIPELINE_STAGE_RESULTS):
        try:
            from utils.pipeline_bridge import hydrate_pipeline_session_from_output
            hydrate_pipeline_session_from_output(st.session_state[SessionKeys.AUTONOMOUS_RESULT])
        except Exception as exc:
            import traceback
            traceback.print_exc()
            # Surface the error in session state for diagnostics in the UI
            st.session_state[SessionKeys.PIPELINE_ERROR] = str(exc)
def _resolve_canonical_dataframe():
    """Resolve dataframe aliases into a single canonical dataframe or None.

    Canonical preference order:
    1. `SessionKeys.DF` (primary canonical store)
    2. `SessionKeys.UPLOADED_DF` (compatibility)
    3. legacy literal `'df'`
    """
    df = st.session_state.get(SessionKeys.DF)
    if df is None:
        df = st.session_state.get(SessionKeys.UPLOADED_DF)
    if df is None and 'df' in st.session_state:
        df = st.session_state.get('df')
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return None
    return df


def _sync_df_aliases():
    """Ensure all alias keys reflect the canonical dataframe and update the loaded flag."""
    resolved = _resolve_canonical_dataframe()
    if resolved is not None:
        # Write canonical value back to all known aliases for backward compatibility
        st.session_state[SessionKeys.DF] = resolved
        st.session_state[SessionKeys.UPLOADED_DF] = resolved
        st.session_state['df'] = resolved
    else:
        # Remove stale aliases when no canonical dataframe exists
        for k in (SessionKeys.DF, SessionKeys.UPLOADED_DF, 'df'):
            st.session_state.pop(k, None)
    # Always set the dataset loaded flag from the canonical resolved value
    st.session_state[SessionKeys.DATASET_LOADED] = bool(resolved is not None and not getattr(resolved, "empty", True))


def get_dataframe():
    _sync_df_aliases()
    return st.session_state.get(SessionKeys.DF)


def set_dataframe(df, name=None):
    if df is not None and not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
    st.session_state[SessionKeys.DF] = df
    st.session_state[SessionKeys.UPLOADED_DF] = df
    st.session_state['df'] = df
    st.session_state[SessionKeys.DATASET_LOADED] = bool(df is not None and not getattr(df, "empty", True))
    if name:
        st.session_state[SessionKeys.DATASET_NAME] = str(name)
        st.session_state[SessionKeys.UPLOAD_FILENAME] = str(name)
    if 'cleaning_issues' in st.session_state and isinstance(st.session_state.cleaning_issues, list) and len(st.session_state.cleaning_issues) > 0:
        norm = []
        for i in st.session_state.cleaning_issues:
            if isinstance(i, dict):
                norm.append(i)
            else:
                norm.append({'title': 'Data Quality Metric', 'description': str(i), 'recommendation': 'Cleaned automatically.', 'severity': 'medium'})
        st.session_state.cleaning_issues = norm

def has_dataset():
    df = get_dataframe()
    return df is not None and not getattr(df, "empty", True)

def get_dataset_name():
    return st.session_state.get(SessionKeys.DATASET_NAME) or st.session_state.get(SessionKeys.UPLOAD_FILENAME) or 'Untitled Dataset'
def get_metadata():
    _sync_df_aliases()
    return st.session_state.get(SessionKeys.DATASET_METADATA) or {}


def normalize_problem_type(problem_type):
    if not isinstance(problem_type, str):
        return "Classification"
    normalized = problem_type.strip().lower()
    if "class" in normalized:
        return "Classification"
    if "regress" in normalized:
        return "Regression"
    return "Classification"


def get_problem_type(output=None) -> str:
    """Single source of truth for detected ML task type across UI and reports."""
    cached = st.session_state.get(SessionKeys.PROBLEM_TYPE)
    if cached:
        return normalize_problem_type(cached)

    if output is None:
        output = get_autonomous_result()
    if isinstance(output, dict):
        dataset_report = output.get("dataset_report") or output.get("dataset_analysis") or {}
        if isinstance(dataset_report, dict):
            problem_analysis = coalesce_dict(dataset_report.get("problem_analysis"))
            pt = problem_analysis.get("problem_type")
            if pt:
                return normalize_problem_type(str(pt))
        artifacts = output.get("training_artifacts") or {}
        if isinstance(artifacts, dict) and artifacts.get("problem_type"):
            return normalize_problem_type(artifacts["problem_type"])
        model_results = output.get("model_results") or {}
        if isinstance(model_results, dict) and model_results.get("problem_type"):
            return normalize_problem_type(model_results["problem_type"])

    metadata = get_metadata()
    analysis = metadata.get("analysis") if isinstance(metadata, dict) else {}
    if isinstance(analysis, dict) and analysis.get("problem_type"):
        return normalize_problem_type(str(analysis["problem_type"]))

    return "Classification"


def get_autonomous_result():
    return st.session_state.get(SessionKeys.AUTONOMOUS_RESULT)

def set_autonomous_result(result):
    st.session_state[SessionKeys.AUTONOMOUS_RESULT] = result

def has_autonomous_result():
    return bool(get_autonomous_result())
def reset_model_pipeline():
    st.session_state.model_trained = False
    st.session_state.report_generated = False
def reset_eda_pipeline():
    st.session_state.eda_generated = False
def persist_dataset_metadata(df):
    """Store dataset analysis and health metadata for UI pages."""
    if df is None or getattr(df, "empty", True):
        st.session_state[SessionKeys.DATASET_METADATA] = {}
        return

    try:
        from agents.dataset_agent import analyze_dataset
        from utils.health_score import compute_health_score

        health = compute_health_score(df)
        analysis = analyze_dataset(df)
        st.session_state[SessionKeys.DATASET_METADATA] = {
            "analysis": analysis,
            "health": health,
            "health_score": health.get("score"),
        }
    except Exception:
        st.session_state[SessionKeys.DATASET_METADATA] = {}


def reset_on_new_dataset(df, name):
    set_dataframe(df, name)
    reset_model_pipeline()
    reset_eda_pipeline()
    persist_dataset_metadata(df)
    st.session_state[SessionKeys.AUTONOMOUS_RESULT] = None
    st.session_state[SessionKeys.PIPELINE_STAGE_RESULTS] = {}
    st.session_state[SessionKeys.PIPELINE_STAGE_STATUSES] = {}
    st.session_state[SessionKeys.PIPELINE_COMPLETED_STAGES] = []
    st.session_state[SessionKeys.PIPELINE_PROGRESS] = 0
    st.session_state[SessionKeys.PIPELINE_RUNNING] = False
    st.session_state[SessionKeys.PIPELINE_EXECUTED] = False
    st.session_state[SessionKeys.REPORT_GENERATED] = False
    st.session_state[SessionKeys.REPORT_PATH] = None
    st.session_state[SessionKeys.REPORT_PAYLOAD] = None
    st.session_state.pop("_applied_pipeline_result_id", None)
    st.session_state.pop("pipeline_validation_warnings", None)
    st.session_state[SessionKeys.PROBLEM_TYPE] = None
    st.session_state[SessionKeys.EXECUTIVE_METRICS] = {}
    st.session_state[SessionKeys.CONFIDENCE_SCORE] = None
def store_model_results(results, best_name, best_model, problem_type, X, y, extras=None):
    st.session_state[SessionKeys.MODEL_TRAINED] = True
    st.session_state[SessionKeys.RESULTS] = results
    st.session_state[SessionKeys.BEST_MODEL_NAME] = best_name
    st.session_state[SessionKeys.BEST_MODEL] = best_model
    st.session_state[SessionKeys.PROBLEM_TYPE] = normalize_problem_type(problem_type)
    if best_name and isinstance(results, dict) and best_name in results:
        st.session_state[SessionKeys.BEST_SCORE] = results[best_name]
    if X is not None:
        st.session_state[SessionKeys.X_DATA] = pd.DataFrame(X)
    st.session_state[SessionKeys.Y_DATA] = y
    if isinstance(extras, dict):
        if extras.get("detailed_metrics"):
            st.session_state[SessionKeys.MODEL_METRICS] = extras["detailed_metrics"]
        if extras.get("model_comparison"):
            st.session_state[SessionKeys.MODEL_LEADERBOARD] = extras["model_comparison"]
        if extras.get("training_times"):
            st.session_state.setdefault("training_times", extras["training_times"])
    st.session_state[SessionKeys.SHAP_COMPUTED] = False
    st.session_state[SessionKeys.REPORT_GENERATED] = False

def store_model_results_with_encoder(results, best_name, best_model, problem_type, X, y, encoder=None, extras=None):
    store_model_results(results, best_name, best_model, problem_type, X, y, extras=extras)
    if encoder is not None:
        st.session_state[SessionKeys.TARGET_ENCODER] = encoder


def navigate_to_view(view_key: str) -> None:
    """Switch enterprise navigation view and rerun."""
    st.session_state["enterprise_nav_view"] = view_key
    st.rerun()


def get_session_health(df=None, target_column=None):
    """Return health score from session cache — never recompute if profile exists."""
    profile = st.session_state.get(SessionKeys.ANALYSIS_PROFILE) or {}
    cached = profile.get("health")
    if isinstance(cached, dict) and cached.get("score") is not None:
        return cached

    meta = get_metadata()
    health = (meta or {}).get("health")
    if isinstance(health, dict) and health.get("score") is not None:
        return health

    if df is None:
        df = get_dataframe()
    if df is not None and not df.empty:
        try:
            from utils.performance_cache import get_cached_health_score
            return get_cached_health_score(df, target_column=target_column or st.session_state.get(SessionKeys.TARGET_COLUMN))
        except Exception:
            from utils.health_score import compute_health_score
            return compute_health_score(df, target_column=target_column)
    return {"score": 0, "grade": "N/A", "letter_grade": "N/A", "recommendations": []}


def get_pipeline_snapshot():
    """Unified read-only snapshot for all UI pages."""
    output = get_autonomous_result() if has_autonomous_result() else None
    return {
        "output": output,
        "df": get_dataframe() if has_dataset() else None,
        "dataset_name": get_dataset_name() if has_dataset() else None,
        "target_column": st.session_state.get(SessionKeys.TARGET_COLUMN),
        "problem_type": st.session_state.get(SessionKeys.PROBLEM_TYPE),
        "best_model": st.session_state.get(SessionKeys.BEST_MODEL_NAME),
        "best_score": st.session_state.get(SessionKeys.BEST_SCORE),
        "results": st.session_state.get(SessionKeys.RESULTS) or {},
        "model_metrics": st.session_state.get(SessionKeys.MODEL_METRICS) or {},
        "leaderboard": st.session_state.get(SessionKeys.MODEL_LEADERBOARD) or [],
        "eda_insights": st.session_state.get(SessionKeys.EDA_INSIGHTS) or [],
        "recommendations": st.session_state.get(SessionKeys.RECOMMENDATIONS) or [],
        "shap_importance": st.session_state.get(SessionKeys.SHAP_IMPORTANCE),
        "confidence_score": st.session_state.get(SessionKeys.CONFIDENCE_SCORE),
        "validation_report": st.session_state.get(SessionKeys.VALIDATION_REPORT),
        "analysis_profile": st.session_state.get(SessionKeys.ANALYSIS_PROFILE) or {},
    }
