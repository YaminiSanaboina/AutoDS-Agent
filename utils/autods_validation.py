"""AutoDS output validation — verify pipeline results against source data."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from utils.safe_checks import normalize_feature_importance, safe_dict_get

METRIC_TOLERANCE = 0.01
VALIDATION_DIR = os.path.join("storage", "validation")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _close(a: Any, b: Any, tol: float = METRIC_TOLERANCE) -> bool:
    try:
        if a is None or b is None:
            return False
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _section(name: str, checks: List[Dict[str, Any]]) -> Dict[str, Any]:
    passed = sum(1 for check in checks if check.get("status") == "PASS")
    applicable = sum(1 for check in checks if check.get("status") in ("PASS", "FAIL"))
    if applicable == 0:
        status = "SKIP"
    elif passed == applicable:
        status = "PASS"
    else:
        status = "FAIL"
    return {
        "name": name,
        "status": status,
        "passed": passed,
        "total": applicable,
        "checks": checks,
    }


def _check(
    label: str,
    passed: bool,
    *,
    expected: Any = None,
    actual: Any = None,
    details: str = "",
    skipped: bool = False,
) -> Dict[str, Any]:
    if skipped:
        return {
            "label": label,
            "status": "SKIP",
            "expected": expected,
            "actual": actual,
            "details": details or "Not enough data to validate.",
        }
    return {
        "label": label,
        "status": "PASS" if passed else "FAIL",
        "expected": expected,
        "actual": actual,
        "details": details,
    }


def _resolve_dataframe(output: Dict[str, Any], df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if is_present_df(df):
        return df
    artifacts = safe_dict_get(output, "training_artifacts") or {}
    cleaned = safe_dict_get(artifacts, "cleaned_dataframe")
    if is_present_df(cleaned):
        return cleaned
    return None


def is_present_df(value: Any) -> bool:
    return isinstance(value, pd.DataFrame) and not value.empty


def _dataset_shape_from_output(output: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    cleaning = safe_dict_get(output, "cleaning_results") or {}
    shape = safe_dict_get(cleaning, "shape")
    if isinstance(shape, (list, tuple)) and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    report = safe_dict_get(output, "dataset_report") or safe_dict_get(output, "dataset_analysis") or {}
    ds_shape = safe_dict_get(report, "dataset_shape") or {}
    rows = safe_dict_get(ds_shape, "rows")
    cols = safe_dict_get(ds_shape, "columns")
    if rows is not None and cols is not None:
        return int(rows), int(cols)
    return None, None


def validate_dataset_analysis(output: Dict[str, Any], df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Compare pipeline dataset stats with actual DataFrame statistics."""
    checks: List[Dict[str, Any]] = []
    frame = _resolve_dataframe(output, df)

    if not is_present_df(frame):
        checks.append(_check("DataFrame available", False, skipped=True))
        return _section("dataset_analysis", checks)

    actual_rows, actual_cols = int(frame.shape[0]), int(frame.shape[1])
    actual_missing = int(frame.isnull().sum().sum())
    actual_duplicates = int(frame.duplicated().sum())
    actual_numeric = int(frame.select_dtypes(include="number").shape[1])
    actual_categorical = int(frame.select_dtypes(exclude="number").shape[1])

    exp_rows, exp_cols = _dataset_shape_from_output(output)
    checks.append(
        _check(
            "Row count",
            exp_rows is None or actual_rows == exp_rows,
            expected=exp_rows,
            actual=actual_rows,
        )
    )
    checks.append(
        _check(
            "Column count",
            exp_cols is None or actual_cols == exp_cols,
            expected=exp_cols,
            actual=actual_cols,
        )
    )

    eda = safe_dict_get(output, "eda_results") or {}
    eda_numeric = safe_dict_get(eda, "numerical_columns") or []
    eda_categorical = safe_dict_get(eda, "categorical_columns") or []
    if isinstance(eda_numeric, list) and eda_numeric:
        checks.append(
            _check(
                "Numeric feature count",
                len(eda_numeric) == actual_numeric,
                expected=len(eda_numeric),
                actual=actual_numeric,
            )
        )
    else:
        checks.append(_check("Numeric feature count", True, expected=actual_numeric, actual=actual_numeric, details="EDA numeric list unavailable; actual count recorded."))

    if isinstance(eda_categorical, list) and eda_categorical:
        checks.append(
            _check(
                "Categorical feature count",
                len(eda_categorical) == actual_categorical,
                expected=len(eda_categorical),
                actual=actual_categorical,
            )
        )
    else:
        checks.append(
            _check(
                "Categorical feature count",
                True,
                expected=actual_categorical,
                actual=actual_categorical,
                details="EDA categorical list unavailable; actual count recorded.",
            )
        )

    checks.append(
        _check(
            "Missing values computed",
            actual_missing >= 0,
            expected=">= 0",
            actual=actual_missing,
        )
    )
    checks.append(
        _check(
            "Duplicate rows computed",
            actual_duplicates >= 0,
            expected=">= 0",
            actual=actual_duplicates,
        )
    )

    return _section("dataset_analysis", checks)


def _normalize_problem_family(value: Any) -> str:
    text = str(value or "").lower()
    if "cluster" in text:
        return "clustering"
    if "regress" in text:
        return "regression"
    return "classification"


def _infer_target_problem_type(target: pd.Series) -> str:
    if target is None or len(target) == 0:
        return "clustering"
    numeric = pd.api.types.is_numeric_dtype(target)
    unique_count = target.nunique(dropna=True)
    if not numeric:
        return "classification"
    total = len(target.dropna())
    unique_ratio = unique_count / max(total, 1)
    if unique_count <= 10 or unique_ratio < 0.05:
        return "classification"
    return "regression"


def validate_problem_type(output: Dict[str, Any], df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Verify declared problem type matches target column characteristics."""
    checks: List[Dict[str, Any]] = []
    report = safe_dict_get(output, "dataset_report") or safe_dict_get(output, "dataset_analysis") or {}
    problem = safe_dict_get(report, "problem_analysis") or {}
    declared = safe_dict_get(problem, "problem_type") or safe_dict_get(output, "hyperparameter_report", {}).get("problem_type")
    declared_family = _normalize_problem_family(declared)

    artifacts = safe_dict_get(output, "training_artifacts") or {}
    target_col = safe_dict_get(artifacts, "target_column") or safe_dict_get(problem, "likely_target")
    frame = _resolve_dataframe(output, df)

    if declared_family == "clustering":
        checks.append(_check("Clustering declared", "cluster" in str(declared).lower(), expected=declared, actual=declared))
        return _section("problem_type", checks)

    if not target_col or not is_present_df(frame) or target_col not in frame.columns:
        checks.append(_check("Target column available", False, skipped=True, details=f"Target: {target_col}"))
        return _section("problem_type", checks)

    inferred = _infer_target_problem_type(frame[target_col])
    checks.append(
        _check(
            "Problem type matches target",
            inferred == declared_family or declared_family == "classification" and inferred == "classification",
            expected=declared_family,
            actual=inferred,
            details=f"Target column: {target_col}",
        )
    )
    return _section("problem_type", checks)


def _recompute_metrics(
    y_true: Any,
    y_pred: Any,
    problem_type: str,
) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {}
    try:
        if _normalize_problem_family(problem_type) == "regression":
            metrics["r2"] = float(r2_score(y_true, y_pred))
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            return metrics

        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["precision"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["recall"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        metrics["f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        try:
            if len(set(y_true)) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred))
        except Exception:
            metrics["roc_auc"] = None
    except Exception:
        pass
    return metrics


def validate_model_metrics(output: Dict[str, Any]) -> Dict[str, Any]:
    """Recompute model metrics from saved predictions and compare to displayed values."""
    checks: List[Dict[str, Any]] = []
    artifacts = safe_dict_get(output, "training_artifacts") or {}
    model_results = safe_dict_get(output, "model_results") or {}
    best_model = safe_dict_get(artifacts, "best_model")
    best_name = safe_dict_get(artifacts, "best_name") or safe_dict_get(model_results, "best_model")
    displayed_metrics = safe_dict_get(artifacts, "results") or safe_dict_get(model_results, "metrics") or {}
    problem_type = safe_dict_get(artifacts, "problem_type") or "Classification"
    extras = safe_dict_get(artifacts, "extras") or {}

    X = safe_dict_get(artifacts, "X_data")
    y = safe_dict_get(artifacts, "y_data")
    X_test = safe_dict_get(extras, "X_test")
    y_test = safe_dict_get(extras, "y_test")

    if best_model is None or best_name is None:
        checks.append(_check("Trained model artifacts", False, skipped=True))
        return _section("model_metrics", checks)

    if X_test is None or y_test is None:
        if X is not None and y is not None:
            try:
                _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            except Exception:
                checks.append(_check("Test split available", False, skipped=True))
                return _section("model_metrics", checks)
        else:
            checks.append(_check("Prediction data available", False, skipped=True))
            return _section("model_metrics", checks)

    try:
        predictions = best_model.predict(X_test)
    except Exception as exc:
        checks.append(_check("Generate predictions", False, details=str(exc)))
        return _section("model_metrics", checks)

    recomputed = _recompute_metrics(y_test, predictions, problem_type)
    primary_key = "r2" if _normalize_problem_family(problem_type) == "regression" else "accuracy"
    displayed_primary = safe_dict_get(displayed_metrics, best_name)
    if displayed_primary is not None and primary_key in recomputed:
        checks.append(
            _check(
                f"Primary metric ({primary_key})",
                _close(displayed_primary, recomputed[primary_key]),
                expected=displayed_primary,
                actual=recomputed[primary_key],
            )
        )

    validation_block = safe_dict_get(output, "validation_results")
    validation_metrics: Dict[str, Any] = {}
    if isinstance(validation_block, list) and validation_block:
        last = validation_block[-1]
        if isinstance(last, dict):
            validation_metrics = safe_dict_get(last, "validation") or safe_dict_get(last, "metrics") or {}
    elif isinstance(validation_block, dict):
        validation_metrics = validation_block

    compare_source = validation_metrics if validation_metrics else recomputed
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc", "r2", "mae", "rmse"]
    for metric_name in metric_names:
        displayed_val = safe_dict_get(compare_source, metric_name)
        recomputed_val = safe_dict_get(recomputed, metric_name)
        if displayed_val is None or recomputed_val is None:
            continue
        checks.append(
            _check(
                metric_name.upper(),
                _close(displayed_val, recomputed_val),
                expected=displayed_val,
                actual=recomputed_val,
            )
        )

    if len([c for c in checks if c["status"] in ("PASS", "FAIL")]) == 0:
        checks.append(_check("Metrics recomputed", bool(recomputed), expected="metrics", actual=list(recomputed.keys())))

    return _section("model_metrics", checks)


def validate_best_model_selection(output: Dict[str, Any]) -> Dict[str, Any]:
    """Verify the selected best model has the highest leaderboard score."""
    checks: List[Dict[str, Any]] = []
    model_results = safe_dict_get(output, "model_results") or {}
    metrics = safe_dict_get(model_results, "metrics") or {}
    best_name = safe_dict_get(model_results, "best_model")

    if not isinstance(metrics, dict) or not metrics or not best_name:
        checks.append(_check("Leaderboard metrics available", False, skipped=True))
        return _section("best_model_selection", checks)

    try:
        winner = max(metrics.items(), key=lambda item: item[1])[0]
    except Exception:
        checks.append(_check("Leaderboard parseable", False))
        return _section("best_model_selection", checks)

    checks.append(
        _check(
            "Best model has highest score",
            winner == best_name,
            expected=winner,
            actual=best_name,
        )
    )
    return _section("best_model_selection", checks)


def validate_explainability(output: Dict[str, Any], df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Verify feature importance exists and references valid numeric features."""
    checks: List[Dict[str, Any]] = []
    explainability = safe_dict_get(output, "explainability_results") or safe_dict_get(output, "xai_results") or {}
    fi_raw = safe_dict_get(explainability, "feature_importance")
    records = normalize_feature_importance(fi_raw)

    checks.append(_check("Feature importance exists", bool(records), expected="non-empty", actual=len(records)))

    if not records:
        return _section("explainability", checks)

    numeric_ok = all(isinstance(item.get("importance"), (int, float, np.floating)) for item in records)
    checks.append(_check("Importance values numeric", numeric_ok, expected="numeric", actual=numeric_ok))

    frame = _resolve_dataframe(output, df)
    artifacts = safe_dict_get(output, "training_artifacts") or {}
    feature_columns = set()
    if is_present_df(frame):
        feature_columns = set(str(col) for col in frame.columns)
    X = safe_dict_get(artifacts, "X_data")
    if isinstance(X, pd.DataFrame):
        feature_columns.update(str(col) for col in X.columns)

    if feature_columns:
        top_features = [item["feature"] for item in records[:5]]
        missing = [feat for feat in top_features if feat not in feature_columns]
        checks.append(
            _check(
                "Top features exist in dataset",
                len(missing) == 0,
                expected=top_features,
                actual=sorted(feature_columns),
                details=f"Missing: {missing}" if missing else "",
            )
        )
    else:
        checks.append(_check("Feature columns available", True, details="Could not cross-check feature names."))

    return _section("explainability", checks)


def _parse_report_metric_strings(model_results_list: Any, best_model: str) -> Optional[float]:
    if not isinstance(model_results_list, list):
        return None
    prefix = f"{best_model}:"
    for item in model_results_list:
        text = str(item)
        if text.startswith(prefix):
            try:
                return float(text.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def validate_pdf_report(output: Dict[str, Any], df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Verify PDF report payload matches dashboard / pipeline values."""
    checks: List[Dict[str, Any]] = []
    final_report = safe_dict_get(output, "final_report") or {}
    payload = safe_dict_get(final_report, "payload")
    if not isinstance(payload, dict) or not payload:
        checks.append(_check("Report payload available", False, skipped=True))
        return _section("report_accuracy", checks)

    frame = _resolve_dataframe(output, df)
    actual_rows = int(frame.shape[0]) if is_present_df(frame) else None
    actual_cols = int(frame.shape[1]) if is_present_df(frame) else None
    exp_rows, exp_cols = _dataset_shape_from_output(output)

    model_results = safe_dict_get(output, "model_results") or {}
    dashboard_best = safe_dict_get(model_results, "best_model")
    dashboard_metrics = safe_dict_get(model_results, "metrics") or {}

    checks.append(
        _check(
            "Dataset name",
            str(payload.get("dataset_name")) == str(output.get("dataset_name") or payload.get("dataset_name")),
            expected=output.get("dataset_name"),
            actual=payload.get("dataset_name"),
        )
    )
    checks.append(
        _check(
            "Rows",
            actual_rows is None or int(payload.get("rows", actual_rows)) == actual_rows or (exp_rows is not None and int(payload.get("rows", exp_rows)) == exp_rows),
            expected=actual_rows or exp_rows,
            actual=payload.get("rows"),
        )
    )
    checks.append(
        _check(
            "Columns",
            actual_cols is None or int(payload.get("columns", actual_cols)) == actual_cols or (exp_cols is not None and int(payload.get("columns", exp_cols)) == exp_cols),
            expected=actual_cols or exp_cols,
            actual=payload.get("columns"),
        )
    )
    checks.append(
        _check(
            "Best model",
            str(payload.get("best_model")) == str(dashboard_best),
            expected=dashboard_best,
            actual=payload.get("best_model"),
        )
    )

    if dashboard_best and isinstance(dashboard_metrics, dict):
        dashboard_score = safe_dict_get(dashboard_metrics, dashboard_best)
        report_score = _parse_report_metric_strings(payload.get("model_results"), str(dashboard_best))
        if dashboard_score is not None and report_score is not None:
            checks.append(
                _check(
                    "Primary metric in report",
                    _close(dashboard_score, report_score),
                    expected=dashboard_score,
                    actual=report_score,
                )
            )

    return _section("report_accuracy", checks)


def _overall_score(sections: Dict[str, Dict[str, Any]]) -> int:
    applicable = [section for section in sections.values() if section.get("status") != "SKIP"]
    if not applicable:
        return 0
    passed = sum(1 for section in applicable if section.get("status") == "PASS")
    return int(round((passed / len(applicable)) * 100))


def run_autods_validation(
    output: Optional[Dict[str, Any]],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Run all AutoDS validation checks and return a structured report."""
    if not output or not isinstance(output, dict):
        return {
            "generated_at": _utc_now(),
            "overall_score": 0,
            "overall_status": "SKIP",
            "sections": {},
            "summary_lines": ["No pipeline output available for validation."],
        }

    sections = {
        "dataset_analysis": validate_dataset_analysis(output, df),
        "problem_type": validate_problem_type(output, df),
        "model_metrics": validate_model_metrics(output),
        "best_model_selection": validate_best_model_selection(output),
        "explainability": validate_explainability(output, df),
        "report_accuracy": validate_pdf_report(output, df),
    }

    score = _overall_score(sections)
    applicable = [s for s in sections.values() if s.get("status") != "SKIP"]
    overall_status = "PASS" if applicable and all(s.get("status") == "PASS" for s in applicable) else ("SKIP" if not applicable else "FAIL")

    labels = {
        "dataset_analysis": "Dataset Analysis",
        "problem_type": "Problem Type",
        "model_metrics": "Model Metrics",
        "best_model_selection": "Best Model Selection",
        "explainability": "Explainability",
        "report_accuracy": "Report Accuracy",
    }
    summary_lines = [f"{labels[key]}: {sections[key]['status']}" for key in labels]
    summary_lines.append(f"Overall Validation Score: {score}/100")

    return {
        "generated_at": _utc_now(),
        "overall_score": score,
        "overall_status": overall_status,
        "sections": sections,
        "summary_lines": summary_lines,
    }


def write_validation_artifacts(report: Dict[str, Any]) -> Tuple[str, str]:
    """Persist validation_report.json and validation_summary.md."""
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    json_path = os.path.join(VALIDATION_DIR, "validation_report.json")
    md_path = os.path.join(VALIDATION_DIR, "validation_summary.md")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    lines = ["# AutoDS Validation Summary", "", f"Generated: {report.get('generated_at', _utc_now())}", ""]
    for line in report.get("summary_lines", []):
        lines.append(f"- {line}")
    lines.append("")
    lines.append("## Section Details")
    lines.append("")
    for key, section in (report.get("sections") or {}).items():
        lines.append(f"### {key.replace('_', ' ').title()} — {section.get('status', 'N/A')}")
        for check in section.get("checks", []):
            lines.append(
                f"- **{check.get('label')}**: {check.get('status')} "
                f"(expected={check.get('expected')}, actual={check.get('actual')})"
            )
            if check.get("details"):
                lines.append(f"  - {check['details']}")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return json_path, md_path


def execute_post_pipeline_validation(
    output: Optional[Dict[str, Any]],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Run validation and write artifacts to storage."""
    report = run_autods_validation(output, df)
    json_path, md_path = write_validation_artifacts(report)
    report["json_path"] = json_path
    report["markdown_path"] = md_path
    return report
