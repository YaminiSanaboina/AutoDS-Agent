"""Bridge MasterAutonomousPipeline outputs to Streamlit session state."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from utils.autods_validation import execute_post_pipeline_validation
from utils.safe_checks import coalesce_dict, is_present, safe_dict_get, coerce_numeric_score, resolve_canonical_accuracy, format_accuracy_display
from utils.data_consistency_validator import DataConsistencyValidator

from agents.dataset_agent import analyze_dataset
from utils.ai_insights import generate_model_insights
from utils.health_score import compute_health_score, detect_data_issues
from utils.session_manager import (
    SessionKeys,
    get_dataframe,
    get_dataset_name,
    normalize_problem_type,
    persist_dataset_metadata,
    set_autonomous_result,
    set_dataframe,
    store_model_results,
)

_logger = logging.getLogger(__name__)

_STAGE_OUTPUT_KEYS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("dataset_intelligence", ("dataset_report", "dataset_analysis", "dataset_profile")),
    ("data_cleaning", ("cleaning_results",)),
    ("eda", ("eda_results",)),
    ("feature_engineering", ("feature_engineering_results",)),
    ("automl", ("model_results", "automl_results")),
    ("model_comparison", ("model_comparison",)),
    ("explainability", ("explainability_results", "xai_results", "shap_results")),
    ("ai_ethics_trust", ("ai_trust_results", "ethics_report", "trust_results")),
    ("deployment_readiness", ("deployment_readiness", "deployment_results")),
    ("self_improvement", ("improvement_history",)),
    ("pdf_report", ("final_report", "executive_report")),
)

_REQUIRED_OUTPUT_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("dataset profile", ("dataset_report", "dataset_analysis", "dataset_profile")),
    ("model results", ("model_results", "automl_results")),
)


def _first_present(output: Dict[str, Any], keys: Tuple[str, ...]) -> Any:
    for key in keys:
        val = output.get(key)
        if val is None:
            continue
        if isinstance(val, (dict, list)) and not val:
            continue
        return val
    return None


def build_stage_results_from_output(output: Dict[str, Any]) -> Dict[str, Any]:
    """Map autonomous pipeline output to UI stage result keys."""
    if not isinstance(output, dict):
        return {}
    results: Dict[str, Any] = {}
    for stage_key, output_keys in _STAGE_OUTPUT_KEYS:
        val = _first_present(output, output_keys)
        if val is not None:
            results[stage_key] = val
    recommendation = output.get("recommendation")
    if recommendation:
        results["ai_decision"] = recommendation
    documentation = output.get("documentation")
    if is_present(documentation):
        results.setdefault("ai_decision", documentation)
    return results


def build_stage_statuses_from_results(stage_results: Dict[str, Any]) -> Dict[str, str]:
    return {stage: "completed" for stage in stage_results}


def validate_pipeline_output(output: Dict[str, Any]) -> List[str]:
    """Return human-readable warnings for missing required pipeline sections."""
    if not isinstance(output, dict):
        return ["Pipeline output is missing or invalid."]
    missing: List[str] = []
    for label, keys in _REQUIRED_OUTPUT_GROUPS:
        if _first_present(output, keys) is None:
            missing.append(label)
    return missing


def hydrate_pipeline_session_from_output(output: Dict[str, Any]) -> None:
    """Restore stage results and statuses from a persisted autonomous result."""
    stage_results = build_stage_results_from_output(output)
    if stage_results:
        st.session_state[SessionKeys.PIPELINE_STAGE_RESULTS] = stage_results
        st.session_state[SessionKeys.PIPELINE_STAGE_STATUSES] = build_stage_statuses_from_results(stage_results)
        st.session_state[SessionKeys.PIPELINE_COMPLETED_STAGES] = list(stage_results.keys())
        st.session_state[SessionKeys.PIPELINE_PROGRESS] = 100
        st.session_state[SessionKeys.PIPELINE_RUNNING] = False


def normalize_pipeline_output(output: Dict[str, Any]) -> Dict[str, Any]:
    """Add UI-compatible alias keys without removing existing pipeline keys."""
    if not isinstance(output, dict):
        return {}

    normalized = dict(output)

    dataset_analysis = _first_present(normalized, ("dataset_analysis", "dataset_report", "dataset_profile"))
    if is_present(dataset_analysis):
        normalized["dataset_analysis"] = dataset_analysis
        normalized["dataset_report"] = dataset_analysis
        normalized["dataset_profile"] = dataset_analysis

    explainability = _first_present(normalized, ("explainability_results", "xai_results", "shap_results"))
    if is_present(explainability):
        normalized["explainability_results"] = explainability
        normalized["xai_results"] = explainability
        normalized["shap_results"] = explainability

    model_results = _first_present(normalized, ("model_results", "automl_results"))
    if is_present(model_results):
        normalized["model_results"] = model_results
        normalized["automl_results"] = model_results

    trust = _first_present(normalized, ("ai_trust_results", "ethics_report", "trust_results"))
    if is_present(trust):
        normalized["ai_trust_results"] = trust
        normalized["trust_results"] = trust

    deploy = _first_present(normalized, ("deployment_readiness", "deployment_results"))
    if is_present(deploy):
        normalized["deployment_readiness"] = deploy
        normalized["deployment_results"] = deploy

    final_report = _first_present(normalized, ("final_report", "executive_report"))
    if is_present(final_report):
        normalized["final_report"] = final_report
        normalized["executive_report"] = final_report
        if isinstance(final_report, dict):
            normalized["report_path"] = final_report.get("path")
            normalized["report_payload"] = final_report.get("payload")
            normalized["final_report_path"] = final_report.get("path")
            normalized["final_report_payload"] = final_report.get("payload")

    model_results = _first_present(normalized, ("model_results", "automl_results"))
    if is_present(model_results) and isinstance(model_results, dict):
        best_model = safe_dict_get(model_results, "best_model")
        if best_model is not None:
            normalized["best_model"] = best_model

    trust_block = _first_present(normalized, ("ai_trust_results", "ethics_report", "trust_results"))
    if is_present(trust_block) and isinstance(trust_block, dict):
        trust_score = safe_dict_get(trust_block, "trust_score")
        if trust_score is not None:
            normalized["trust_score"] = trust_score
        if normalized.get("confidence_score") is None:
            normalized["confidence_score"] = normalized.get("final_ai_confidence_score") or trust_score

    deploy = _first_present(normalized, ("deployment_readiness", "deployment_results"))
    if is_present(deploy) and isinstance(deploy, dict):
        normalized["deployment_status"] = deploy.get("risk_level") or deploy.get("status") or deploy.get("deployment_status")
        if deploy.get("risk_level") is not None:
            normalized["deployment_risk"] = deploy.get("risk_level")
        elif deploy.get("risk") is not None:
            normalized["deployment_risk"] = deploy.get("risk")

    if normalized.get("final_score") is None:
        overall_score = coalesce_dict(normalized.get("final_scores", {})).get("overall_score")
        if overall_score is not None:
            normalized["final_score"] = overall_score

    return normalized


def _parse_feature_importance(raw_value: Any):
    if raw_value is None:
        return None
    if isinstance(raw_value, pd.DataFrame):
        return raw_value
    if isinstance(raw_value, dict):
        return pd.DataFrame(list(raw_value.items()), columns=["Feature", "Importance"]).sort_values(
            "Importance", ascending=False
        )
    return None


def _build_recovery_messages(stage_errors: List[Dict[str, Any]]) -> List[str]:
    messages: List[str] = []
    if not stage_errors:
        return messages

    try:
        from agents.self_healing_agent import SelfHealingAgent

        healer = SelfHealingAgent()
    except Exception:
        healer = None

    for entry in stage_errors:
        stage = entry.get("stage", "unknown")
        error_text = entry.get("error", "Unknown error")
        if entry.get("user_message"):
            messages.append(f"{stage}: {entry['user_message']}")
            continue
        if healer is not None:
            try:
                analysis = healer.analyze_error(error_text)
                fix = healer.recommend_fix(analysis)
                explanation = fix.get("explanation") or analysis.get("root_cause") or error_text
                action = fix.get("recommended_action")
                if action:
                    messages.append(f"{stage}: {explanation} Recommended action: {action}")
                else:
                    messages.append(f"{stage}: {explanation}")
                continue
            except Exception:
                pass
        messages.append(f"{stage}: {error_text}")

    return messages


def _build_model_leaderboard(
    results: Dict[str, Any],
    detailed_metrics: Dict[str, Any],
    training_times: Dict[str, Any],
    problem_type: str = "Classification",
) -> List[Dict[str, Any]]:
    """Build per-model AutoML leaderboard rows from training metrics (not HPO iterations)."""
    from utils.model_ranking import rank_models_by_composite

    if not detailed_metrics:
        return []

    ranked = rank_models_by_composite(detailed_metrics, problem_type, results)
    return [
        {
            "model": name,
            "metrics": detailed_metrics.get(name, {}),
            "training_time": training_times.get(name),
            "composite_score": score,
        }
        for name, score in ranked
    ]


def _ensure_executive_metrics(output: Dict[str, Any]) -> Dict[str, Any]:
    """Build or repair executive metrics — single source of truth for UI and exports."""
    from agents.trust_score_calculator import TrustScoreCalculator, create_executive_metrics_object

    existing = coalesce_dict(output.get("executive_metrics"))
    model_results = coalesce_dict(output.get("model_results"))
    deployment = coalesce_dict(output.get("deployment_readiness"))
    trust_block = coalesce_dict(output.get("ai_trust_results"))
    final_scores = coalesce_dict(output.get("final_scores"))
    dataset_score = final_scores.get("dataset_score")
    if dataset_score is None and isinstance(output.get("dataset_report"), dict):
        dataset_score = coalesce_dict(output["dataset_report"].get("intelligence_score")).get("score", 0)

    best_name = model_results.get("best_model") or existing.get("best_model")
    if existing.get("best_model") and existing.get("best_model") != "—" and float(existing.get("trust_score") or 0) > 0:
        resolved_acc = resolve_canonical_accuracy(existing, model_results, best_model=best_name)
        if resolved_acc is not None:
            if 0 < resolved_acc <= 1:
                existing["accuracy"] = round(resolved_acc * 100, 2)
            else:
                existing["accuracy"] = round(resolved_acc, 2)
        existing["trust_score"] = round(float(existing["trust_score"]), 1)
        if best_name and existing.get("best_model") in (None, "—"):
            existing["best_model"] = best_name
        return existing

    trust_raw = existing.get("trust_score") or trust_block.get("trust_score") or output.get("final_ai_confidence_score")
    trust_numeric = coerce_numeric_score(trust_raw)
    if trust_numeric is None or trust_numeric <= 0:
        perf = resolve_canonical_accuracy(None, model_results, best_model=best_name)
        trust_numeric = TrustScoreCalculator.calculate_trust_score(
            dataset_health=dataset_score,
            fairness_score=trust_block.get("fairness_score"),
            deployment_readiness=deployment,
            model_results=model_results,
            explainability_available=is_present(
                coalesce_dict(output.get("explainability_results")).get("feature_importance")
            ),
            model_performance=perf,
        )

    resolved_accuracy = resolve_canonical_accuracy(existing, model_results, best_model=best_name)
    final_score = output.get("final_score") or final_scores.get("overall_score") or 0
    risk = deployment.get("risk_level", "Unknown")
    if final_score >= 80 and str(risk).lower() == "low":
        deploy_status = "Production Ready"
    elif final_score >= 60 or str(risk).lower() == "medium":
        deploy_status = "Needs Monitoring"
    else:
        deploy_status = "Not Ready"

    return create_executive_metrics_object(
        best_model=best_name,
        accuracy=resolved_accuracy,
        trust_score=trust_numeric,
        risk_level=risk,
        deployment_status=deploy_status,
        deployment_readiness=deployment,
        health_score=dataset_score,
        confidence_score=trust_numeric,
        final_decision={
            "risk_level": risk,
            "deployment_status": deploy_status,
            "trust_score": trust_numeric,
            "recommendation": output.get("recommendation"),
        },
        runtime_seconds=output.get("total_runtime"),
        model_results=model_results,
        deployment_info=deployment,
    )


def _finalize_executive_metrics_display(
    executive_metrics: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    """Attach canonical accuracy_display for Home, Reports, PDF, and Excel."""
    model_results = coalesce_dict(output.get("model_results"))
    problem_type = normalize_problem_type(
        st.session_state.get(SessionKeys.PROBLEM_TYPE)
        or safe_dict_get(coalesce_dict(safe_dict_get(output.get("dataset_report"), "problem_analysis")), "problem_type")
        or "Classification"
    )
    canonical = resolve_canonical_accuracy(
        executive_metrics,
        model_results,
        best_model=executive_metrics.get("best_model"),
        session_best_score=st.session_state.get(SessionKeys.BEST_SCORE),
    )
    if canonical is not None:
        if 0 < canonical <= 1:
            executive_metrics["accuracy"] = round(canonical * 100, 2)
        else:
            executive_metrics["accuracy"] = round(canonical, 2)
    executive_metrics["accuracy_display"] = format_accuracy_display(
        executive_metrics.get("accuracy") or canonical,
        problem_type,
    )
    return executive_metrics


def _inject_regression_metrics(
    executive_metrics: Dict[str, Any],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    """For regression problems, extract r2/rmse/mae/mse into executive_metrics from model_comparison or validation_results."""
    problem_type = executive_metrics.get("problem_type") or safe_dict_get(
        coalesce_dict(safe_dict_get(output.get("dataset_report"), "problem_analysis")),
        "problem_type",
    ) or "Classification"
    
    # Only inject for regression
    if "regress" not in str(problem_type).lower():
        return executive_metrics
    
    # Extract from model_comparison if available
    model_comparison = output.get("model_comparison")
    if isinstance(model_comparison, list):
        for entry in model_comparison:
            if isinstance(entry, dict):
                metrics = entry.get("metrics", {})
                if isinstance(metrics, dict):
                    for key in ("r2", "rmse", "mae", "mse"):
                        if key in metrics and key not in executive_metrics:
                            executive_metrics[key] = metrics[key]
    
    # Extract from validation_results as fallback
    validation_results = output.get("validation_results")
    if isinstance(validation_results, list):
        for entry in validation_results:
            if isinstance(entry, dict):
                validation = entry.get("validation", {})
                if isinstance(validation, dict):
                    for key in ("r2", "rmse", "mae", "mse"):
                        if key in validation and key not in executive_metrics:
                            executive_metrics[key] = validation[key]
                    break
    
    return executive_metrics


def _sync_report_payload_with_executive_metrics(
    output: Dict[str, Any],
    executive_metrics: Dict[str, Any],
) -> None:
    """Align PDF payload fields with executive metrics (single source of truth)."""
    final_report = coalesce_dict(output.get("final_report"))
    report_payload = final_report.get("payload") if isinstance(final_report.get("payload"), dict) else None
    if not isinstance(report_payload, dict):
        return

    accuracy_display = executive_metrics.get("accuracy_display") or format_accuracy_display(
        executive_metrics.get("accuracy"),
        normalize_problem_type(st.session_state.get(SessionKeys.PROBLEM_TYPE) or "Classification"),
    )
    problem_type_value = normalize_problem_type(
        st.session_state.get(SessionKeys.PROBLEM_TYPE)
        or executive_metrics.get("problem_type")
        or safe_dict_get(coalesce_dict(output.get("model_results")), "problem_type")
        or "Classification"
    )
    synced_payload = {
        **report_payload,
        "accuracy": executive_metrics.get("accuracy"),
        "accuracy_display": accuracy_display,
        "problem_type": problem_type_value,
        "best_model": executive_metrics.get("best_model") or report_payload.get("best_model"),
        "trust_score": executive_metrics.get("trust_score"),
        "deployment_status": executive_metrics.get("deployment_status"),
        "deployment_readiness_score": safe_dict_get(
            coalesce_dict(executive_metrics.get("deployment_decision")),
            "readiness_score",
        ),
    }
    model_results = coalesce_dict(output.get("model_results"))
    if model_results:
        if not synced_payload.get("detailed_metrics"):
            synced_payload["detailed_metrics"] = coalesce_dict(safe_dict_get(model_results, "detailed_metrics"))
        if not synced_payload.get("training_times"):
            synced_payload["training_times"] = coalesce_dict(safe_dict_get(model_results, "training_times"))
        if not synced_payload.get("results_metrics"):
            synced_payload["results_metrics"] = coalesce_dict(safe_dict_get(model_results, "metrics"))
        if not synced_payload.get("model_selection_notes"):
            from utils.model_ranking import build_best_model_consistency_notes

            synced_payload["model_selection_notes"] = build_best_model_consistency_notes(
                synced_payload.get("best_model"),
                coalesce_dict(synced_payload.get("detailed_metrics")),
                problem_type_value,
                synced_payload.get("results_metrics"),
                coalesce_dict(safe_dict_get(model_results, "model_selection_explanation")),
            )
        if not synced_payload.get("uploaded_file"):
            synced_payload["uploaded_file"] = report_payload.get("uploaded_file") or output.get("dataset_name")
        if not synced_payload.get("detected_dataset"):
            synced_payload["detected_dataset"] = report_payload.get("detected_dataset")

    from utils.pdf_visualizations import (
        chart_prefix_from_name,
        ensure_eda_chart_paths,
        ensure_explainability_chart_paths,
    )

    chart_prefix = chart_prefix_from_name(
        synced_payload.get("uploaded_file") or output.get("dataset_name")
    )
    if not synced_payload.get("eda_chart_paths"):
        chart_df = None
        artifacts = coalesce_dict(output.get("training_artifacts"))
        chart_df = artifacts.get("cleaned_dataframe")
        if not isinstance(chart_df, pd.DataFrame):
            chart_df = get_dataframe()
        synced_payload["eda_chart_paths"] = ensure_eda_chart_paths(
            chart_df if isinstance(chart_df, pd.DataFrame) else None,
            synced_payload.get("target_column"),
            output.get("eda_results"),
            chart_prefix,
            existing_paths=report_payload.get("eda_chart_paths") if isinstance(report_payload, dict) else None,
        )
    if not synced_payload.get("explainability_chart_paths"):
        synced_payload["explainability_chart_paths"] = ensure_explainability_chart_paths(
            output.get("explainability_results"),
            chart_prefix,
            existing_paths=report_payload.get("explainability_chart_paths") if isinstance(report_payload, dict) else None,
        )

    report_path = final_report.get("path")
    needs_pdf_regeneration = not report_path or (
        report_payload.get("accuracy_display") != synced_payload.get("accuracy_display")
        or normalize_problem_type(report_payload.get("problem_type") or "") != problem_type_value
    )
    if needs_pdf_regeneration:
        try:
            from agents.report_agent import generate_pdf_report

            report_path = generate_pdf_report(synced_payload)
        except Exception as exc:
            _logger.warning("PDF regeneration with synced metrics failed: %s", exc)
            report_path = final_report.get("path")

    output["final_report"] = {
        **final_report,
        "payload": synced_payload,
        "path": report_path,
    }
    st.session_state[SessionKeys.REPORT_PAYLOAD] = synced_payload
    if report_path:
        st.session_state[SessionKeys.REPORT_PATH] = report_path
        st.session_state[SessionKeys.REPORT_GENERATED] = True


def apply_autonomous_result_to_session(output: Optional[Dict[str, Any]]) -> None:
    """Persist pipeline output into shared session state for all UI pages."""
    if not output or not isinstance(output, dict):
        st.session_state["pipeline_validation_warnings"] = ["Pipeline produced no result object."]
        _logger.warning("apply_autonomous_result_to_session called with empty output")
        return

    output = normalize_pipeline_output(output)
    warnings = validate_pipeline_output(output)
    st.session_state["pipeline_validation_warnings"] = warnings
    if warnings:
        _logger.warning("Pipeline validation warnings: %s", warnings)

    set_autonomous_result(output)
    hydrate_pipeline_session_from_output(output)
    
    # Ensure completion state is persisted: progress=100, running=False
    # This guarantees Home page and Reports show final state without navigation.
    st.session_state[SessionKeys.PIPELINE_PROGRESS] = 100
    st.session_state[SessionKeys.PIPELINE_RUNNING] = False

    dataset_name = output.get("dataset_name") or get_dataset_name()
    artifacts = output.get("training_artifacts") or {}

    cleaned_df = artifacts.get("cleaned_dataframe")
    if isinstance(cleaned_df, pd.DataFrame) and not cleaned_df.empty:
        set_dataframe(cleaned_df, dataset_name)
    elif get_dataframe() is not None:
        persist_dataset_metadata(get_dataframe())

    dataset_report = output.get("dataset_report") or output.get("dataset_analysis") or {}
    meta = st.session_state.get(SessionKeys.DATASET_METADATA) or {}
    try:
        basic_analysis = analyze_dataset(get_dataframe()) if get_dataframe() is not None else {}
    except Exception:
        basic_analysis = meta.get("analysis") or {}
    st.session_state[SessionKeys.DATASET_METADATA] = {
        **meta,
        "analysis": basic_analysis or meta.get("analysis", {}),
        "dataset_report": dataset_report,
        "intelligence_report": dataset_report,
        "health": meta.get("health") or (compute_health_score(get_dataframe()) if get_dataframe() is not None else {}),
    }

    target_column = artifacts.get("target_column")
    if not target_column and isinstance(dataset_report, dict):
        problem_analysis = coalesce_dict(dataset_report.get("problem_analysis"))
        target_column = problem_analysis.get("likely_target")
    if target_column:
        st.session_state[SessionKeys.TARGET_COLUMN] = target_column

    cleaning_results = output.get("cleaning_results") or {}
    cleaning_report = cleaning_results.get("report")
    if cleaning_report is not None:
        st.session_state[SessionKeys.CLEANING_REPORT] = cleaning_report
    if get_dataframe() is not None:
        try:
            st.session_state[SessionKeys.CLEANING_ISSUES] = detect_data_issues(get_dataframe())
        except Exception:
            pass

    eda_results = output.get("eda_results") or {}
    if eda_results:
        st.session_state[SessionKeys.EDA_GENERATED] = True
        st.session_state[SessionKeys.EDA_SUMMARY] = eda_results.get("summary")
        st.session_state[SessionKeys.EDA_NUMERICAL_COLUMNS] = eda_results.get("numerical_columns", [])
        st.session_state[SessionKeys.EDA_CATEGORICAL_COLUMNS] = eda_results.get("categorical_columns", [])
        st.session_state[SessionKeys.EDA_INSIGHTS] = eda_results.get("insights", [])

    best_model = artifacts.get("best_model")
    hyperparameter_report = coalesce_dict(output.get("hyperparameter_report"))
    problem_type = artifacts.get("problem_type") or hyperparameter_report.get("problem_type")
    if not problem_type and isinstance(dataset_report, dict):
        problem_type = coalesce_dict(dataset_report.get("problem_analysis")).get("problem_type")
    X = artifacts.get("X_data")
    y = artifacts.get("y_data")

    model_results = coalesce_dict(safe_dict_get(output, "model_results"))
    best_name = artifacts.get("best_name") or safe_dict_get(model_results, "best_model")
    results = artifacts.get("results") or safe_dict_get(model_results, "metrics") or {}
    detailed_metrics = coalesce_dict(safe_dict_get(model_results, "detailed_metrics"))
    training_times = coalesce_dict(safe_dict_get(model_results, "training_times"))
    training_extras = coalesce_dict(artifacts.get("extras"))
    model_leaderboard = _build_model_leaderboard(
        results,
        detailed_metrics,
        training_times,
        normalize_problem_type(problem_type or "Classification"),
    )

    if best_name and results:
        store_model_results(
            results,
            best_name,
            best_model if best_model is not None else best_name,
            normalize_problem_type(problem_type or "Classification"),
            X,
            y,
            extras={
                "detailed_metrics": detailed_metrics,
                "model_comparison": model_leaderboard,
                "training_times": training_times,
            },
        )
    elif best_name:
        st.session_state[SessionKeys.BEST_MODEL_NAME] = best_name
        st.session_state[SessionKeys.MODEL_TRAINED] = True
        if model_results.get("best_score") is not None:
            st.session_state[SessionKeys.BEST_SCORE] = model_results.get("best_score")
        if results:
            st.session_state[SessionKeys.RESULTS] = results
        if problem_type:
            st.session_state[SessionKeys.PROBLEM_TYPE] = normalize_problem_type(problem_type)

    encoder = artifacts.get("target_encoder")
    if best_name and (best_model is not None or results):
        if encoder is not None:
            st.session_state[SessionKeys.TARGET_ENCODER] = encoder

        if training_extras.get("confusion_matrix") is not None:
            st.session_state[SessionKeys.CONFUSION_MATRIX] = training_extras["confusion_matrix"]
        if training_extras.get("roc_data") is not None:
            st.session_state[SessionKeys.ROC_DATA] = training_extras["roc_data"]
        if is_present(training_extras.get("feature_importance")):
            parsed_fi = _parse_feature_importance(training_extras["feature_importance"])
            if parsed_fi is not None and not getattr(parsed_fi, "empty", True):
                st.session_state[SessionKeys.FEATURE_IMPORTANCE] = parsed_fi

    explainability = output.get("explainability_results") or {}
    feature_importance = _parse_feature_importance(explainability.get("feature_importance"))
    shap_values = explainability.get("shap_values")

    if feature_importance is not None and not getattr(feature_importance, "empty", True):
        st.session_state[SessionKeys.SHAP_COMPUTED] = True
        st.session_state[SessionKeys.SHAP_IMPORTANCE] = feature_importance
        st.session_state[SessionKeys.FEATURE_IMPORTANCE] = feature_importance
        if shap_values is not None:
            st.session_state[SessionKeys.SHAP_VALUES] = shap_values

        if get_dataframe() is not None and best_name and results:
            try:
                insights, recommendations, confidence = generate_model_insights(
                    get_dataframe(),
                    st.session_state.get(SessionKeys.PROBLEM_TYPE, "Classification"),
                    best_name,
                    results,
                    feature_importance,
                    st.session_state.get(SessionKeys.TARGET_COLUMN),
                )
                st.session_state[SessionKeys.AI_INSIGHTS] = insights
                st.session_state[SessionKeys.RECOMMENDATIONS] = recommendations
                st.session_state[SessionKeys.CONFIDENCE_SCORE] = confidence
            except Exception:
                pass

    trust_block = coalesce_dict(output.get("ai_trust_results"))
    executive_metrics = _ensure_executive_metrics(output)
    executive_metrics = _finalize_executive_metrics_display(executive_metrics, output)
    executive_metrics = _inject_regression_metrics(executive_metrics, output)
    output["executive_metrics"] = executive_metrics
    trust_numeric = executive_metrics.get("trust_score")
    if trust_numeric is not None:
        st.session_state[SessionKeys.CONFIDENCE_SCORE] = trust_numeric
    if trust_block:
        trust_block["trust_score"] = trust_numeric
        output["ai_trust_results"] = trust_block
    # Ensure deployment_readiness always has risk_level set
    deployment_readiness = output.get("deployment_readiness") or {}
    if not isinstance(deployment_readiness, dict):
        deployment_readiness = {}
    if "risk_level" not in deployment_readiness or not deployment_readiness.get("risk_level"):
        deployment_readiness["risk_level"] = "Unknown"
    output["deployment_readiness"] = deployment_readiness

    _sync_report_payload_with_executive_metrics(output, executive_metrics)

    stage_errors = output.get("stage_errors") or []
    recovery_messages = _build_recovery_messages(stage_errors)
    st.session_state["pipeline_recovery_messages"] = recovery_messages
    st.session_state[SessionKeys.PIPELINE_EXECUTED] = True

    df_now = get_dataframe()
    if df_now is not None:
        target = st.session_state.get(SessionKeys.TARGET_COLUMN)
        st.session_state[SessionKeys.ANALYSIS_PROFILE] = {
            "health": compute_health_score(df_now, target_column=target),
            "eda_summary": st.session_state.get(SessionKeys.EDA_SUMMARY),
            "best_model": st.session_state.get(SessionKeys.BEST_MODEL_NAME),
            "best_score": st.session_state.get(SessionKeys.BEST_SCORE),
        }

    try:
        validation_report = execute_post_pipeline_validation(output, get_dataframe())
        st.session_state[SessionKeys.VALIDATION_REPORT] = validation_report
        st.session_state[SessionKeys.VALIDATION_SCORE] = validation_report.get("overall_score")
    except Exception:
        st.session_state[SessionKeys.VALIDATION_REPORT] = None
        st.session_state[SessionKeys.VALIDATION_SCORE] = None

    st.session_state["_pipeline_just_finished"] = True

    try:
        st.session_state[SessionKeys.EXECUTIVE_METRICS] = executive_metrics
        st.session_state[SessionKeys.PIPELINE_COMPLETE] = True
        st.session_state[SessionKeys.PIPELINE_CURRENT_STAGE] = "pdf_report"
        is_valid = DataConsistencyValidator.validate_all(executive_metrics)
        if not is_valid:
            _logger.warning("[PipelineBridge] Executive metrics validation found issues")
    except Exception as exc:
        _logger.error("[PipelineBridge] Failed to persist executive metrics: %s", exc)
        st.session_state[SessionKeys.EXECUTIVE_METRICS] = executive_metrics or {}
