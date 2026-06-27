"""Safe truthiness helpers for pandas and other ambiguous values."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def is_present(value: Any) -> bool:
    """Return True when value contains meaningful content without ambiguous DataFrame checks."""
    if value is None:
        return False
    if isinstance(value, pd.DataFrame):
        return not value.empty
    if isinstance(value, pd.Series):
        return not value.empty
    if hasattr(value, "empty"):
        try:
            return not value.empty
        except Exception:
            pass
    if isinstance(value, (str, bytes)):
        return bool(value.strip()) if isinstance(value, str) else bool(value)
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) > 0
    try:
        return len(value) > 0
    except Exception:
        return True


def safe_bool(value: Any, default: bool = False) -> bool:
    """Convert a value to bool without raising on pandas objects."""
    try:
        return is_present(value)
    except Exception:
        return default


def coalesce_dict(value: Any) -> Dict[str, Any]:
    """Return a dict without using `value or {}` (unsafe for DataFrames)."""
    if isinstance(value, dict):
        return value
    return {}


def coalesce_list(value: Any) -> List[Any]:
    """Return a list without using `value or []` (unsafe for DataFrames/Series)."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def normalize_recommendations(value: Any) -> List[str]:
    """Normalize recommendations to a deduplicated list of strings."""
    if value is None:
        return []
    if isinstance(value, dict):
        flattened: List[str] = []
        for key, items in value.items():
            if isinstance(items, list):
                flattened.extend(str(item) for item in items)
            else:
                flattened.append(f"{key}: {items}")
        value = flattened
    elif isinstance(value, tuple):
        value = [str(item) for item in value]
    elif isinstance(value, list):
        value = [str(item) for item in value]
    else:
        value = [str(value)]

    deduped: List[str] = []
    seen = set()
    for item in value:
        normalized = " ".join(str(item).strip().split())
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _feature_column_name(columns: List[str]) -> Optional[str]:
    lowered = {str(col).lower(): col for col in columns}
    for candidate in ("feature", "features", "name", "variable"):
        if candidate in lowered:
            return str(lowered[candidate])
    return str(columns[0]) if columns else None


def _importance_column_name(columns: List[str]) -> Optional[str]:
    lowered = {str(col).lower(): col for col in columns}
    for candidate in ("importance", "mean |shap|", "shap", "value", "score", "weight"):
        if candidate in lowered:
            return str(lowered[candidate])
    return str(columns[-1]) if columns else None


def normalize_feature_importance(value: Any) -> List[Dict[str, Any]]:
    """Normalize SHAP / feature importance into chart-ready records."""
    if value is None:
        return []

    records: List[Dict[str, Any]] = []

    if isinstance(value, pd.DataFrame):
        if value.empty:
            return []
        feat_col = _feature_column_name(list(value.columns))
        imp_col = _importance_column_name(list(value.columns))
        if feat_col is None or imp_col is None:
            return []
        for _, row in value.iterrows():
            try:
                records.append({"feature": str(row[feat_col]), "importance": float(row[imp_col])})
            except (TypeError, ValueError):
                continue
        return records

    if isinstance(value, pd.Series):
        if value.empty:
            return []
        for feature, importance in value.items():
            try:
                records.append({"feature": str(feature), "importance": float(importance)})
            except (TypeError, ValueError):
                continue
        return records

    if isinstance(value, dict):
        for feature, importance in value.items():
            try:
                records.append({"feature": str(feature), "importance": float(importance)})
            except (TypeError, ValueError):
                continue
        return records

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                feature = item.get("feature") or item.get("Feature") or item.get("name")
                importance = item.get("importance") or item.get("Importance") or item.get("value")
                if feature is not None and importance is not None:
                    try:
                        records.append({"feature": str(feature), "importance": float(importance)})
                    except (TypeError, ValueError):
                        continue
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    records.append({"feature": str(item[0]), "importance": float(item[1])})
                except (TypeError, ValueError):
                    continue
        return records

    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                for idx, importance in enumerate(value.tolist()):
                    try:
                        records.append({"feature": f"Feature {idx + 1}", "importance": float(importance)})
                    except (TypeError, ValueError):
                        continue
            return records
    except Exception:
        pass

    return records


def feature_importance_as_dict(value: Any) -> Dict[str, float]:
    """Convert normalized importance records to a dict for legacy consumers."""
    return {item["feature"]: item["importance"] for item in normalize_feature_importance(value)}


def safe_dict_get(mapping: Any, key: str, default: Any = None) -> Any:
    """Dict .get without treating a DataFrame mapping as truthy."""
    if not isinstance(mapping, dict):
        return default
    return mapping.get(key, default)


def first_present(*values: Any) -> Any:
    """Return the first value that is meaningfully present (pandas-safe)."""
    for value in values:
        if is_present(value):
            return value
    return None


def coerce_numeric_score(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Parse trust/fairness scores without raising on 'Not Evaluated' or blanks."""
    if value is None or value == "":
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"not evaluated", "n/a", "na", "—", "-", "none", "unknown"}:
            return default
        try:
            return float(value.strip())
        except ValueError:
            return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def format_score_display(value: Any, *, suffix: str = "", unavailable: str = "Not Evaluated") -> str:
    """Human-readable score for UI metrics and gauges."""
    numeric = coerce_numeric_score(value)
    if numeric is not None:
        if suffix == "%":
            return f"{numeric:.0f}%"
        if suffix:
            return f"{numeric:.0f}{suffix}"
        return f"{numeric:.0f}"
    if isinstance(value, str) and value.strip():
        return value.strip()
    return unavailable


def resolve_canonical_accuracy(
    executive_metrics: Optional[Dict[str, Any]] = None,
    model_results: Optional[Dict[str, Any]] = None,
    *,
    best_model: Optional[str] = None,
    session_best_score: Any = None,
) -> Optional[float]:
    """Resolve the best model score from executive_metrics or model_results."""
    em = coalesce_dict(executive_metrics)
    mr = coalesce_dict(model_results)
    name = best_model or em.get("best_model") or mr.get("best_model")

    em_acc = em.get("accuracy")
    if em_acc is not None:
        try:
            val = float(em_acc)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass

    for candidate in (
        mr.get("best_score"),
        session_best_score,
        safe_dict_get(coalesce_dict(mr.get("metrics")), name) if name else None,
    ):
        if candidate is None:
            continue
        try:
            val = float(candidate)
            if val > 0:
                return val
        except (TypeError, ValueError):
            continue
    return None


def format_accuracy_display(
    value: Any,
    problem_type: str = "Classification",
    *,
    unavailable: str = "Unavailable",
) -> str:
    """Format model accuracy for UI, PDF, and Excel. Accepts 0–1 decimal or 0–100 percent."""
    if value is None or value == "":
        return unavailable
    try:
        score = float(value)
    except (TypeError, ValueError):
        text = str(value).strip()
        return text if text else unavailable
    if score <= 0:
        return unavailable
    is_regression = "regress" in str(problem_type).lower()
    if is_regression:
        if score > 1 and score <= 100:
            score = score / 100.0
        return f"R² {score:.4f}"
    if score <= 1:
        return f"{score * 100:.1f}%"
    return f"{score:.1f}%"


def display_kpi_value(value: Any, *, unavailable: str = "Unavailable") -> str:
    """Safe KPI string — never show None, empty, or bare dashes."""
    if value is None:
        return unavailable
    if isinstance(value, str):
        text = value.strip()
        if not text or text in ("—", "-", "None", "null", "N/A", "n/a"):
            return unavailable
        return text
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).strip()
    return text if text else unavailable
