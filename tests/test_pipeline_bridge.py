"""Tests for pipeline output -> session state bridge."""

from __future__ import annotations

import pandas as pd
import pytest
import streamlit as st

from utils.pipeline_bridge import (
    apply_autonomous_result_to_session,
    build_stage_results_from_output,
    build_stage_statuses_from_results,
    normalize_pipeline_output,
    validate_pipeline_output,
)
from utils.session_manager import SessionKeys, init_session, set_dataframe


@pytest.fixture
def session_ready():
    init_session()
    return True


def test_normalize_pipeline_output_adds_aliases():
    output = {
        "dataset_analysis": {"summary": "ok"},
        "explainability_results": {"feature_importance": {"a": 1.0}},
        "model_results": {"best_model": "rf", "metrics": {"rf": 0.9}},
        "ai_trust_results": {"trust_score": 82},
        "deployment_readiness": {"readiness_score": 75},
        "final_report": {"path": "reports/x.pdf"},
        "final_ai_confidence_score": 88,
        "final_scores": {"overall_score": 0.92},
    }
    normalized = normalize_pipeline_output(output)
    assert normalized["dataset_report"] == output["dataset_analysis"]
    assert normalized["dataset_profile"] == output["dataset_analysis"]
    assert normalized["xai_results"] == output["explainability_results"]
    assert normalized["shap_results"] == output["explainability_results"]
    assert normalized["automl_results"] == output["model_results"]
    assert normalized["trust_results"] == output["ai_trust_results"]
    assert normalized["deployment_results"] == output["deployment_readiness"]
    assert normalized["executive_report"] == output["final_report"]
    assert normalized["best_model"] == "rf"
    assert normalized["trust_score"] == 82
    assert normalized["report_path"] == "reports/x.pdf"
    assert normalized["final_report_path"] == "reports/x.pdf"
    assert normalized["report_payload"] is None
    assert normalized["final_report_payload"] is None
    assert normalized["confidence_score"] == 88
    assert normalized["final_score"] == 0.92


def test_build_stage_results_from_output_maps_stages():
    output = {
        "dataset_report": {"rows": 100},
        "cleaning_results": {"report": {}},
        "eda_results": {"insights": ["a"]},
        "model_results": {"best_model": "rf"},
        "explainability_results": {"feature_importance": {"x": 1.0}},
        "ai_trust_results": {"trust_score": 80},
        "deployment_readiness": {"readiness_score": 70},
        "final_report": {"path": "reports/x.pdf"},
    }
    stages = build_stage_results_from_output(output)
    assert "dataset_intelligence" in stages
    assert "data_cleaning" in stages
    assert "eda" in stages
    assert "automl" in stages
    assert "explainability" in stages
    assert "ai_ethics_trust" in stages
    assert "deployment_readiness" in stages
    assert "pdf_report" in stages
    assert build_stage_statuses_from_results(stages)["eda"] == "completed"


def test_validate_pipeline_output_flags_missing_sections():
    warnings = validate_pipeline_output({"eda_results": {}})
    assert "dataset profile" in warnings
    assert "model results" in warnings


def test_format_accuracy_display_percent_and_decimal():
    from utils.safe_checks import format_accuracy_display, resolve_canonical_accuracy

    assert format_accuracy_display(0.92, "Classification") == "92.0%"
    assert format_accuracy_display(92.0, "Classification") == "92.0%"
    assert format_accuracy_display(None, "Classification") == "Unavailable"
    assert format_accuracy_display(0, "Classification") == "Unavailable"

    resolved = resolve_canonical_accuracy(
        {"accuracy": 0.0, "best_model": "RF"},
        {"best_model": "RF", "metrics": {"RF": 0.88}},
    )
    assert resolved == 0.88


def test_finalize_executive_metrics_accuracy_display(session_ready):
    from utils.pipeline_bridge import _finalize_executive_metrics_display
    import streamlit as st

    st.session_state[SessionKeys.PROBLEM_TYPE] = "Classification"
    output = {
        "model_results": {
            "best_model": "Random Forest",
            "metrics": {"Random Forest": 0.91},
        },
        "dataset_report": {"problem_analysis": {"problem_type": "Classification"}},
    }
    metrics = {"best_model": "Random Forest", "accuracy": 0.0, "trust_score": 75.0, "deployment_status": "Needs Monitoring"}
    finalized = _finalize_executive_metrics_display(metrics, output)
    assert finalized["accuracy_display"] == "91.0%"
    assert float(finalized["accuracy"]) == pytest.approx(91.0)


def test_ensure_executive_metrics_builds_trust_score():
    from utils.pipeline_bridge import _ensure_executive_metrics

    output = {
        "model_results": {"best_model": "Random Forest", "best_score": 0.92, "metrics": {"Random Forest": 0.92}},
        "deployment_readiness": {"risk_level": "Low"},
        "ai_trust_results": {"fairness_score": 75},
        "final_scores": {"dataset_score": 85, "overall_score": 82},
        "explainability_results": {"feature_importance": {"a": 1.0}},
    }
    metrics = _ensure_executive_metrics(output)
    assert metrics["best_model"] == "Random Forest"
    assert float(metrics["trust_score"]) > 0
    assert metrics["deployment_status"] in ("Production Ready", "Needs Monitoring", "Not Ready")


def test_apply_autonomous_result_populates_report_and_eda(session_ready):
    df = pd.DataFrame({"feature_a": [1, 2, 3], "target": [0, 1, 0]})
    set_dataframe(df, "demo.csv")
    output = {
        "dataset_name": "demo.csv",
        "dataset_analysis": {"problem_analysis": {"likely_target": "target", "problem_type": "Classification"}},
        "eda_results": {
            "summary": "summary table",
            "numerical_columns": ["feature_a"],
            "categorical_columns": [],
            "insights": ["Insight one"],
        },
        "final_report": {
            "path": "reports/demo.pdf",
            "payload": {"executive_summary": "All good"},
        },
        "stage_errors": [],
    }

    apply_autonomous_result_to_session(output)

    assert st.session_state[SessionKeys.EDA_GENERATED] is True
    assert st.session_state[SessionKeys.REPORT_GENERATED] is True
    assert st.session_state[SessionKeys.REPORT_PAYLOAD]["executive_summary"] == "All good"
    assert "accuracy_display" in st.session_state[SessionKeys.REPORT_PAYLOAD]
    assert st.session_state[SessionKeys.REPORT_PATH].endswith(".pdf")
    assert st.session_state[SessionKeys.TARGET_COLUMN] == "target"
    assert st.session_state[SessionKeys.PIPELINE_STAGE_RESULTS].get("eda") is not None
    assert st.session_state[SessionKeys.PIPELINE_EXECUTED] is True


def test_accuracy_consistent_across_home_reports_exports(session_ready):
    from ui.dashboard import build_dashboard_context
    from ui.interactive_report_center import build_report_context
    from utils.report_exports import build_export_context_from_report_ctx

    output = {
        "dataset_name": "iris.csv",
        "dataset_report": {"problem_analysis": {"problem_type": "Classification", "likely_target": "target"}},
        "model_results": {
            "best_model": "SVM",
            "metrics": {"SVM": 1.0},
            "best_score": 1.0,
        },
        "deployment_readiness": {"risk_level": "Low"},
        "ai_trust_results": {"trust_score": 92},
        "final_scores": {"dataset_score": 85, "overall_score": 88},
        "final_report": {
            "path": "reports/demo.pdf",
            "payload": {"executive_summary": "All good", "best_model": "SVM"},
        },
        "executive_metrics": {"best_model": "SVM", "accuracy": 0.0, "trust_score": 92.0},
    }
    apply_autonomous_result_to_session(output)
    home = build_dashboard_context(output)
    report = build_report_context(output, None)
    export = build_export_context_from_report_ctx(report)
    pdf_payload = st.session_state[SessionKeys.REPORT_PAYLOAD]
    exec_metrics = st.session_state[SessionKeys.EXECUTIVE_METRICS]

    expected = "100.0%"
    assert home["score_display"] == expected
    assert report["accuracy_display"] == expected
    assert export["accuracy_display"] == expected
    assert pdf_payload["accuracy_display"] == expected
    assert exec_metrics["accuracy_display"] == expected
    assert home["score_display"] != "0.0%"


def test_export_context_includes_best_model_details(session_ready):
    from ui.interactive_report_center import build_report_context
    from utils.report_exports import build_export_context_from_report_ctx

    output = {
        "dataset_name": "iris.csv",
        "dataset_report": {"problem_analysis": {"problem_type": "Classification", "likely_target": "target"}},
        "model_results": {
            "best_model": "SVM",
            "metrics": {"SVM": 1.0},
            "detailed_metrics": {"SVM": {"accuracy": 1.0, "cv_score": 1.0, "f1": 1.0}},
            "model_selection_explanation": {"why_chosen": ["Selected for highest cross-validated accuracy."]},
        },
        "deployment_readiness": {"risk_level": "Low"},
        "ai_trust_results": {"trust_score": 92},
        "final_report": {"path": "reports/demo.pdf", "payload": {"executive_summary": "All good", "best_model": "SVM"}},
        "executive_metrics": {"best_model": "SVM", "accuracy": 0.0, "trust_score": 92.0},
    }
    apply_autonomous_result_to_session(output)
    report = build_report_context(output, None)
    export = build_export_context_from_report_ctx(report)

    assert export["best_model"] == "SVM"
    assert export["model_selection_explanation"]["why_chosen"] == ["Selected for highest cross-validated accuracy."]
    assert export["best_model_metrics"]["accuracy"] == 1.0
    assert export["best_model_metrics"]["cv_score"] == 1.0
