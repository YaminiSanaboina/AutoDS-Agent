"""Tests for AutoDS validation framework."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils.autods_validation import (
    METRIC_TOLERANCE,
    run_autods_validation,
    validate_best_model_selection,
    validate_dataset_analysis,
    validate_explainability,
    validate_model_metrics,
    validate_pdf_report,
    validate_problem_type,
    write_validation_artifacts,
)
from utils.pipeline_bridge import apply_autonomous_result_to_session
from utils.session_manager import SessionKeys, init_session, set_dataframe


@pytest.fixture
def iris_frame():
    return pd.DataFrame(
        {
            "sepal_length": [5.1, 4.9, 5.0, 4.7, 5.2, 4.6, 5.3, 4.8, 5.4, 4.5],
            "sepal_width": [3.5, 3.0, 3.2, 3.2, 3.4, 3.1, 3.7, 3.0, 3.9, 2.3],
            "petal_length": [1.4, 1.4, 1.5, 1.3, 1.4, 1.3, 1.5, 1.4, 1.6, 1.5],
            "petal_width": [0.2, 0.2, 0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.4, 0.2],
            "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def trained_output(iris_frame):
    X = iris_frame.drop(columns=["target"])
    y = iris_frame["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    from sklearn.metrics import accuracy_score

    accuracy = float(accuracy_score(y_test, preds))
    decision_tree_score = max(0.0, accuracy - 0.2)
    return {
        "dataset_name": "iris_demo.csv",
        "dataset_report": {
            "dataset_shape": {"rows": 10, "columns": 5},
            "problem_analysis": {"problem_type": "Classification", "likely_target": "target"},
        },
        "cleaning_results": {"shape": [10, 5]},
        "eda_results": {
            "numerical_columns": list(iris_frame.columns),
            "categorical_columns": [],
        },
        "model_results": {
            "best_model": "Logistic Regression",
            "metrics": {
                "Logistic Regression": accuracy,
                "Decision Tree": decision_tree_score,
            },
        },
        "explainability_results": {
            "feature_importance": {
                "petal_length": 0.8,
                "petal_width": 0.2,
                "sepal_length": 0.1,
            }
        },
        "training_artifacts": {
            "best_model": model,
            "best_name": "Logistic Regression",
            "results": {"Logistic Regression": accuracy, "Decision Tree": decision_tree_score},
            "problem_type": "Classification",
            "target_column": "target",
            "X_data": X,
            "y_data": y,
            "extras": {"X_test": X_test, "y_test": y_test},
            "cleaned_dataframe": iris_frame,
        },
        "final_report": {
            "path": "reports/demo.pdf",
            "payload": {
                "dataset_name": "iris_demo.csv",
                "rows": 10,
                "columns": 5,
                "best_model": "Logistic Regression",
                "model_results": [f"Logistic Regression: {accuracy:.4f}"],
            },
        },
        "stage_errors": [],
    }


def test_validate_dataset_analysis_passes(iris_frame, trained_output):
    section = validate_dataset_analysis(trained_output, iris_frame)
    assert section["status"] == "PASS"


def test_validate_problem_type_passes(trained_output, iris_frame):
    section = validate_problem_type(trained_output, iris_frame)
    assert section["status"] == "PASS"


def test_validate_best_model_selection_passes(trained_output):
    section = validate_best_model_selection(trained_output)
    assert section["status"] == "PASS"


def test_validate_model_metrics_passes(trained_output):
    section = validate_model_metrics(trained_output)
    assert section["status"] == "PASS"


def test_validate_explainability_passes(trained_output, iris_frame):
    section = validate_explainability(trained_output, iris_frame)
    assert section["status"] == "PASS"


def test_validate_pdf_report_passes(trained_output, iris_frame):
    section = validate_pdf_report(trained_output, iris_frame)
    assert section["status"] == "PASS"


def test_run_autods_validation_overall_score(trained_output, iris_frame):
    report = run_autods_validation(trained_output, iris_frame)
    assert report["overall_score"] == 100
    assert report["overall_status"] == "PASS"
    assert "Dataset Analysis: PASS" in report["summary_lines"][0]


def test_write_validation_artifacts(tmp_path, trained_output, iris_frame, monkeypatch):
    monkeypatch.setattr("utils.autods_validation.VALIDATION_DIR", str(tmp_path))
    report = run_autods_validation(trained_output, iris_frame)
    json_path, md_path = write_validation_artifacts(report)
    assert os.path.exists(json_path)
    assert os.path.exists(md_path)
    with open(json_path, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["overall_score"] == 100


def test_pipeline_bridge_stores_validation_report(trained_output, iris_frame):
    init_session()
    set_dataframe(iris_frame, "iris_demo.csv")
    apply_autonomous_result_to_session(trained_output)
    import streamlit as st

    assert st.session_state[SessionKeys.VALIDATION_SCORE] == 100
    assert st.session_state[SessionKeys.VALIDATION_REPORT]["overall_status"] == "PASS"


def test_metric_tolerance_constant():
    assert METRIC_TOLERANCE == 0.01
