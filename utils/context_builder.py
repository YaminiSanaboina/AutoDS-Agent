"""Build structured AutoDS context from Streamlit session state."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from utils.health_score import compute_health_score, detect_data_issues
from utils.safe_checks import (
    coalesce_dict,
    coalesce_list,
    feature_importance_as_dict,
    first_present,
    is_present,
    normalize_feature_importance,
    normalize_recommendations,
    safe_dict_get,
)
from utils.session_manager import (
    SessionKeys,
    get_autonomous_result,
    get_dataframe,
    get_metadata,
    get_problem_type,
    has_autonomous_result,
)


def build_context() -> Dict[str, Any]:
    """Collect session state and pipeline output into one assistant context."""
    df = get_dataframe()
    metadata = get_metadata()
    output = get_autonomous_result() if has_autonomous_result() else None

    context: Dict[str, Any] = {
        "dataset": build_dataset_context(df, metadata, output),
        "eda": build_eda_context(output),
        "model": build_model_context(output),
        "feature_importance": build_feature_importance_context(output),
        "shap": build_shap_context(),
        "prediction": build_prediction_context(),
        "report": build_report_context(output),
        "pipeline": build_pipeline_context(output),
        "cleaning": build_cleaning_context(output),
        "feature_engineering": build_feature_engineering_context(output),
        "ethics": build_ethics_context(output),
        "deployment": build_deployment_context(output),
        "documentation": build_documentation_context(output),
    }
    return context


def build_dataset_context(df, metadata, output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    dataset_context: Dict[str, Any] = {
        "name": None,
        "rows": 0,
        "columns": 0,
        "feature_names": [],
        "quality_score": None,
        "missing_values": 0,
        "duplicates": 0,
        "target_column": None,
        "issues": [],
    }

    if df is not None:
        dataset_context["name"] = (
            st.session_state.get(SessionKeys.DATASET_NAME)
            or st.session_state.get(SessionKeys.UPLOAD_FILENAME)
            or (output or {}).get("dataset_name")
        )
        dataset_context["rows"], dataset_context["columns"] = df.shape
        dataset_context["feature_names"] = list(df.columns)
        dataset_context["missing_values"] = int(df.isnull().sum().sum())
        dataset_context["duplicates"] = int(df.duplicated().sum())
        dataset_context["missing_by_column"] = {
            str(col): int(df[col].isnull().sum()) for col in df.columns if df[col].isnull().sum() > 0
        }
        dataset_context["column_dtypes"] = {str(col): str(dtype) for col, dtype in df.dtypes.items()}
        try:
            dataset_context["issues"] = detect_data_issues(df)
        except Exception:
            dataset_context["issues"] = st.session_state.get(SessionKeys.CLEANING_ISSUES) or []

    if metadata:
        health = metadata.get("health") or {}
        dataset_context["quality_score"] = health.get("score") if isinstance(health, dict) else metadata.get("score")
        if dataset_context["quality_score"] is None:
            dataset_context["quality_score"] = metadata.get("health_score")

    if df is not None and dataset_context["quality_score"] is None:
        try:
            from utils.session_manager import get_session_health
            health = get_session_health(df)
            dataset_context["quality_score"] = health.get("score")
            dataset_context["quality_grade"] = health.get("letter_grade") or health.get("grade")
        except Exception:
            pass
    elif df is not None:
        try:
            from utils.session_manager import get_session_health
            health = get_session_health(df)
            dataset_context["quality_grade"] = health.get("letter_grade") or health.get("grade")
        except Exception:
            pass

    dataset_context["target_column"] = st.session_state.get(SessionKeys.TARGET_COLUMN)
    if output:
        report = output.get("dataset_report") or output.get("dataset_analysis") or {}
        if isinstance(report, dict):
            problem = report.get("problem_analysis") or {}
            dataset_context["problem_type"] = get_problem_type(output)
            dataset_context["target_column"] = dataset_context["target_column"] or problem.get("likely_target")
            summary = first_present(problem.get("summary"), report.get("summary"))
            if isinstance(summary, str):
                dataset_context["intelligence_summary"] = summary

    return dataset_context


def build_eda_context(output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = {
        "generated": st.session_state.get(SessionKeys.EDA_GENERATED, False),
        "summary": st.session_state.get(SessionKeys.EDA_SUMMARY),
        "numerical_columns": st.session_state.get(SessionKeys.EDA_NUMERICAL_COLUMNS),
        "categorical_columns": st.session_state.get(SessionKeys.EDA_CATEGORICAL_COLUMNS),
        "selected_feature": st.session_state.get(SessionKeys.EDA_SELECTED_FEATURE),
        "selected_feature_2": st.session_state.get(SessionKeys.EDA_SELECTED_FEATURE_2),
        "chart_type": st.session_state.get(SessionKeys.EDA_CHART_TYPE),
        "active_tab": st.session_state.get(SessionKeys.EDA_ACTIVE_TAB),
        "insights": st.session_state.get(SessionKeys.EDA_INSIGHTS),
    }
    if output and output.get("eda_results"):
        eda = output["eda_results"]
        if isinstance(eda, dict):
            ctx["generated"] = True
            ctx["summary"] = first_present(ctx["summary"], eda.get("summary"))
            ctx["insights"] = first_present(ctx["insights"], eda.get("insights"), eda.get("ai_insights"))
            ctx["numerical_columns"] = first_present(ctx["numerical_columns"], eda.get("numerical_columns"))
            ctx["categorical_columns"] = first_present(ctx["categorical_columns"], eda.get("categorical_columns"))
    return ctx


def build_model_context(output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    executive_metrics = st.session_state.get(SessionKeys.EXECUTIVE_METRICS) or {}
    if executive_metrics:
        return {
            "trained": True,
            "best_name": executive_metrics.get("best_model"),
            "best_score": executive_metrics.get("accuracy"),
            "problem_type": get_problem_type(output),
            "score": executive_metrics.get("accuracy"),
            "results": coalesce_dict(safe_dict_get(output, "model_results")) if output else coalesce_dict(st.session_state.get(SessionKeys.RESULTS)),
            "detailed_metrics": coalesce_dict(st.session_state.get(SessionKeys.MODEL_METRICS)),
            "leaderboard": st.session_state.get(SessionKeys.MODEL_LEADERBOARD) or [],
            "confidence_score": executive_metrics.get("confidence_score"),
            "training_times": st.session_state.get("training_times") or {},
        }

    results = coalesce_dict(st.session_state.get(SessionKeys.RESULTS))
    best_name = st.session_state.get(SessionKeys.BEST_MODEL_NAME)
    score = None
    if best_name and best_name in results:
        score = results.get(best_name)

    model_results = coalesce_dict(safe_dict_get(output, "model_results")) if output else {}
    detailed = coalesce_dict(st.session_state.get(SessionKeys.MODEL_METRICS))
    if model_results:
        results = results or coalesce_dict(model_results.get("metrics"))
        best_name = best_name or model_results.get("best_model")
        detailed = detailed or coalesce_dict(model_results.get("detailed_metrics"))
        if best_name and best_name in results:
            score = results.get(best_name)

    return {
        "trained": bool(st.session_state.get(SessionKeys.MODEL_TRAINED, False) or results),
        "best_name": best_name,
        "best_score": st.session_state.get(SessionKeys.BEST_SCORE) or score,
        "problem_type": get_problem_type(output),
        "score": score,
        "results": results,
        "detailed_metrics": detailed,
        "leaderboard": st.session_state.get(SessionKeys.MODEL_LEADERBOARD) or [],
        "confidence_score": st.session_state.get(SessionKeys.CONFIDENCE_SCORE),
        "training_times": st.session_state.get("training_times") or {},
    }


def build_feature_importance_context(output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    importance = st.session_state.get(SessionKeys.FEATURE_IMPORTANCE)
    explainability = coalesce_dict(safe_dict_get(output, "explainability_results"))
    if not explainability:
        explainability = coalesce_dict(safe_dict_get(output, "xai_results"))
    if not is_present(importance) and isinstance(explainability, dict):
        importance = safe_dict_get(explainability, "feature_importance")

    records = normalize_feature_importance(importance)
    top_features = [record["feature"] for record in records[:8]]
    return {"available": bool(top_features), "top_features": top_features}


def build_shap_context() -> Dict[str, Any]:
    return {
        "computed": st.session_state.get(SessionKeys.SHAP_COMPUTED, False),
        "values": st.session_state.get(SessionKeys.SHAP_VALUES),
        "importance": st.session_state.get(SessionKeys.SHAP_IMPORTANCE),
        "positive_negative": st.session_state.get(SessionKeys.SHAP_POSITIVE_NEGATIVE),
    }


def build_prediction_context() -> Dict[str, Any]:
    return {
        "prepared_input": st.session_state.get("prepared_input"),
        "confidence_score": st.session_state.get(SessionKeys.CONFIDENCE_SCORE),
        "last_prediction": st.session_state.get("last_prediction"),
    }


def build_report_context(output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = st.session_state.get(SessionKeys.REPORT_PAYLOAD)
    path = st.session_state.get(SessionKeys.REPORT_PATH)
    generated = st.session_state.get(SessionKeys.REPORT_GENERATED, False)
    if output and output.get("final_report"):
        final = output["final_report"]
        if isinstance(final, dict):
            path = path or final.get("path")
            payload = payload or final.get("payload")
            generated = generated or bool(path or payload)

    executive_metrics = st.session_state.get(SessionKeys.EXECUTIVE_METRICS) or {}
    if isinstance(payload, dict):
        executive = payload.get("executive_summary")
    else:
        executive = None

    deploy = coalesce_dict(safe_dict_get(executive_metrics, "deployment_readiness")) if executive_metrics else {}
    recommendation = safe_dict_get(executive_metrics.get("final_decision") or {}, "recommendation") if executive_metrics else None
    if recommendation is None and output:
        recommendation = output.get("recommendation")
    ethics = coalesce_dict(safe_dict_get(output, "ai_trust_results")) if output else {}
    if not ethics and output:
        ethics = coalesce_dict(safe_dict_get(output, "ethics_report"))
    if not ethics and executive_metrics:
        ethics = coalesce_dict(executive_metrics)

    readiness_score = safe_dict_get(deploy, "readiness_score") or safe_dict_get(deploy, "deployment_score")
    if readiness_score is None and executive_metrics:
        readiness_score = safe_dict_get(executive_metrics.get("deployment_decision") or {}, "readiness_score") or executive_metrics.get("confidence_score")

    return {
        "generated": generated,
        "report_path": path,
        "report_payload": payload,
        "executive_summary": executive,
        "best_model": executive_metrics.get("best_model") if executive_metrics else None,
        "best_score": executive_metrics.get("accuracy") if executive_metrics else None,
        "trust_score": executive_metrics.get("trust_score") if executive_metrics else None,
        "readiness_score": readiness_score,
        "risk_level": executive_metrics.get("risk_level") if executive_metrics else None,
        "deployment_status": executive_metrics.get("deployment_status") if executive_metrics else None,
        "recommendation": recommendation,
        "health_score": executive_metrics.get("health_score") if executive_metrics else None,
        "confidence_score": executive_metrics.get("confidence_score") if executive_metrics else None,
        "ethics": ethics,
        "deploy": deploy,
    }


def build_pipeline_context(output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not output:
        return {
            "executed": st.session_state.get(SessionKeys.PIPELINE_EXECUTED, False),
            "stage_results": st.session_state.get(SessionKeys.PIPELINE_STAGE_RESULTS, {}),
        }

    dataset_report = coalesce_dict(safe_dict_get(output, "dataset_report"))
    if not dataset_report:
        dataset_report = coalesce_dict(safe_dict_get(output, "dataset_analysis"))
    explainability = coalesce_dict(safe_dict_get(output, "explainability_results"))
    if not explainability:
        explainability = coalesce_dict(safe_dict_get(output, "xai_results"))
    fi = feature_importance_as_dict(safe_dict_get(explainability, "feature_importance"))

    documentation = coalesce_dict(safe_dict_get(output, "documentation"))
    sections = safe_dict_get(documentation, "sections")
    if not isinstance(sections, dict):
        sections = {}

    eda_results = coalesce_dict(safe_dict_get(output, "eda_results"))
    deploy = coalesce_dict(safe_dict_get(output, "deployment_readiness"))

    return {
        "executed": True,
        "dataset_name": output.get("dataset_name"),
        "project_goal": output.get("project_goal"),
        "recommendation": output.get("recommendation"),
        "final_score": output.get("final_score"),
        "confidence": output.get("final_ai_confidence_score"),
        "target_column": safe_dict_get(coalesce_dict(safe_dict_get(output, "training_artifacts")), "target_column"),
        "dataset_summary": _extract_summary(dataset_report),
        "eda_insights": safe_dict_get(eda_results, "insights"),
        "model_selection_rationale": sections.get("Model Selection") if isinstance(sections, dict) else None,
        "feature_importance": fi,
        "explainability_summary": safe_dict_get(explainability, "explanation") or safe_dict_get(explainability, "summary"),
        "deployment_risk": safe_dict_get(deploy, "risk_level"),
        "issues": [],
        "stage_results": st.session_state.get(SessionKeys.PIPELINE_STAGE_RESULTS, {}),
    }


def build_cleaning_context(output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    report = st.session_state.get(SessionKeys.CLEANING_REPORT)
    cleaning = (output or {}).get("cleaning_results") or {}
    if isinstance(cleaning, dict):
        report = report or cleaning.get("report")
    actions: List[str] = []
    if isinstance(report, dict):
        actions = coalesce_list(safe_dict_get(report, "actions"))
        if not actions:
            actions = coalesce_list(safe_dict_get(report, "steps"))
    return {"report": report, "actions": actions}


def build_feature_engineering_context(output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    plan = (output or {}).get("feature_engineering_results") or {}
    if not isinstance(plan, dict):
        return {}
    return {
        "steps": coalesce_list(safe_dict_get(plan, "steps")) or coalesce_list(safe_dict_get(plan, "transformations")),
        "recommended_changes": normalize_recommendations(safe_dict_get(plan, "recommended_changes"))
        or normalize_recommendations(safe_dict_get(plan, "recommendations")),
        "created_features": coalesce_list(safe_dict_get(plan, "created_features"))
        or coalesce_list(safe_dict_get(plan, "new_features")),
    }


def build_ethics_context(output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    block: Dict[str, Any] = {}
    if output:
        ethics = output.get("ethics_report")
        trust = output.get("ai_trust_results")
        if isinstance(ethics, dict):
            block = ethics
        elif isinstance(trust, dict):
            block = trust
    bias = coalesce_dict(safe_dict_get(block, "bias_analysis"))
    concerns = normalize_recommendations(safe_dict_get(bias, "bias_concerns"))
    if not concerns:
        concerns = normalize_recommendations(safe_dict_get(block, "concerns"))
    return {
        "trust_score": block.get("trust_score"),
        "fairness_score": block.get("fairness_score"),
        "concerns": concerns,
    }


def build_deployment_context(output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Prefer authoritative executive_metrics when available
    exec_metrics = None
    try:
        import streamlit as st
        from utils.session_manager import SessionKeys

        exec_metrics = st.session_state.get(SessionKeys.EXECUTIVE_METRICS)
    except Exception:
        exec_metrics = None

    if exec_metrics and isinstance(exec_metrics, dict):
        dep = exec_metrics.get("deployment_decision") or {}
        return {
            "risk_level": dep.get("risk_level") or exec_metrics.get("risk_level"),
            "readiness_score": dep.get("readiness_score") or exec_metrics.get("trust_score"),
            "warnings": normalize_recommendations(safe_dict_get(exec_metrics, "deployment_readiness", {}).get("warnings") if isinstance(exec_metrics.get("deployment_readiness"), dict) else []),
            "recommendations": normalize_recommendations(safe_dict_get(exec_metrics.get("deployment_readiness", {}), "recommendations")) or normalize_recommendations(exec_metrics.get("final_decision", {}).get("recommendation") or exec_metrics.get("final_decision", {}).get("recommendation")),
            "recommendation": (exec_metrics.get("final_decision") or {}).get("recommendation"),
        }

    deploy = coalesce_dict(safe_dict_get(output, "deployment_readiness")) if output else {}
    return {
        "risk_level": deploy.get("risk_level"),
        "readiness_score": deploy.get("readiness_score") or deploy.get("deployment_score"),
        "warnings": normalize_recommendations(safe_dict_get(deploy, "warnings")),
        "recommendations": normalize_recommendations(safe_dict_get(deploy, "recommendations"))
        or normalize_recommendations(safe_dict_get(deploy, "api_recommendations")),
        "recommendation": safe_dict_get(output, "recommendation") if output else None,
    }


def build_documentation_context(output: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    doc = (output or {}).get("documentation") or {}
    if not isinstance(doc, dict):
        return {}
    sections = doc.get("sections") or {}
    return {
        "title": doc.get("title"),
        "summary": doc.get("summary"),
        "model_selection": sections.get("Model Selection") if isinstance(sections, dict) else None,
        "deployment": sections.get("Deployment Recommendation") if isinstance(sections, dict) else None,
    }


def _extract_summary(dataset_report: Any) -> Optional[str]:
    if not isinstance(dataset_report, dict):
        return None
    problem = dataset_report.get("problem_analysis") or {}
    summary = problem.get("summary") or dataset_report.get("summary")
    if is_present(summary) and isinstance(summary, str):
        return summary
    return None
