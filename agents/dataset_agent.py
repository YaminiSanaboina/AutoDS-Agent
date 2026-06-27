import html
import re
import warnings
from typing import Any

import pandas as pd

TARGET_KEYWORDS = ("target", "class", "label", "outcome", "churn", "status")
NEXT_STEP = "Go to Data Quality Lab."


def detect_rows_and_columns(df: pd.DataFrame) -> dict[str, int]:
    return {"rows": int(df.shape[0]), "columns": int(df.shape[1])}


def detect_memory_usage(df: pd.DataFrame) -> float:
    return round(df.memory_usage(deep=True).sum() / (1024**2), 2)


def detect_numerical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def detect_categorical_columns(df: pd.DataFrame) -> list[str]:
    numerical = set(detect_numerical_columns(df))
    date_cols = set(detect_date_columns(df))
    return [col for col in df.columns if col not in numerical and col not in date_cols]


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    date_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()

    for col in df.select_dtypes(include=["object", "string"]).columns:
        sample = df[col].dropna().head(25)
        if sample.empty:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            date_cols.append(col)

    return list(dict.fromkeys(date_cols))


def detect_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    missing = df.isnull().sum()
    by_column = {str(col): int(count) for col, count in missing.items() if count > 0}
    return {
        "total": int(missing.sum()),
        "by_column": by_column,
    }


def detect_duplicate_rows(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())


def _column_matches_target_keyword(column_name: str, keyword: str) -> bool:
    name = str(column_name).lower()
    if name == keyword:
        return True
    return re.search(rf"(^|_|-|\b){re.escape(keyword)}($|_|-|\b)", name) is not None


def detect_possible_targets(df: pd.DataFrame) -> list[str]:
    candidates = []
    for col in df.columns:
        if any(_column_matches_target_keyword(col, keyword) for keyword in TARGET_KEYWORDS):
            candidates.append(col)
    return candidates


def detect_problem_type(df: pd.DataFrame, target_column: str | None) -> str:
    if not target_column or target_column not in df.columns:
        return "unknown"

    series = df[target_column]
    if pd.api.types.is_numeric_dtype(series):
        unique_count = series.nunique(dropna=True)
        total = len(series.dropna())
        unique_ratio = unique_count / max(total, 1)
        if unique_count <= 10 or unique_ratio < 0.05:
            return "classification"
        return "regression"

    return "classification"


def detect_class_imbalance(df: pd.DataFrame, target_column: str | None) -> dict[str, Any] | None:
    if not target_column or target_column not in df.columns:
        return None

    if detect_problem_type(df, target_column) != "classification":
        return None

    counts = df[target_column].value_counts(dropna=True)
    if len(counts) < 2:
        return None

    majority = int(counts.iloc[0])
    minority = int(counts.iloc[-1])
    ratio = round(majority / minority, 2) if minority else None

    return {
        "is_imbalanced": ratio is not None and ratio >= 3,
        "majority_class": str(counts.index[0]),
        "minority_class": str(counts.index[-1]),
        "ratio": ratio,
        "class_counts": {str(k): int(v) for k, v in counts.head(10).items()},
    }


def build_ai_recommendations(analysis: dict[str, Any]) -> list[str]:
    recommendations = []
    quality = analysis["data_quality_issues"]
    ml = analysis["ml_understanding"]

    if quality["missing_values"]["total"] > 0:
        recommendations.append(
            "Some columns contain missing values. Review them in Data Quality Lab before modeling."
        )

    if quality["duplicate_rows"] > 0:
        recommendations.append(
            "Duplicate rows were detected. Removing them can improve model reliability."
        )

    imbalance = ml.get("class_imbalance")
    if imbalance and imbalance.get("is_imbalanced"):
        recommendations.append(
            "The target column looks imbalanced. Consider resampling or class-weight strategies later."
        )

    if not recommendations:
        recommendations.append(
            "The dataset looks well structured. A quick quality review is still a good idea."
        )

    return recommendations


def analyze_dataset(df: pd.DataFrame) -> dict[str, Any]:
    shape = detect_rows_and_columns(df)
    numerical = detect_numerical_columns(df)
    categorical = detect_categorical_columns(df)
    date_columns = detect_date_columns(df)
    missing_values = detect_missing_values(df)
    duplicate_rows = detect_duplicate_rows(df)
    possible_targets = detect_possible_targets(df)
    suggested_target = possible_targets[0] if possible_targets else None
    problem_type = detect_problem_type(df, suggested_target)
    class_imbalance = detect_class_imbalance(df, suggested_target)

    structured = {
        "dataset_overview": {
            "rows": shape["rows"],
            "columns": shape["columns"],
            "memory_mb": detect_memory_usage(df),
            "column_names": list(df.columns),
        },
        "column_information": {
            "numerical": numerical,
            "categorical": categorical,
            "date": date_columns,
            "data_types": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        },
        "data_quality_issues": {
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
        },
        "ml_understanding": {
            "possible_targets": possible_targets,
            "suggested_target": suggested_target,
            "problem_type": problem_type,
            "class_imbalance": class_imbalance,
        },
        "next_step": NEXT_STEP,
    }

    structured["ai_recommendations"] = build_ai_recommendations(structured)

    return {
        "rows": shape["rows"],
        "columns": shape["columns"],
        "column_names": list(df.columns),
        "data_types": df.dtypes.astype(str),
        "missing_values": df.isnull().sum(),
        "duplicate_rows": duplicate_rows,
        "summary": df.describe(include="all"),
        **structured,
    }


def format_ai_explanation(analysis: dict[str, Any]) -> str:
    overview = analysis.get("dataset_overview", {})
    columns = analysis.get("column_information", {})
    quality = analysis.get("data_quality_issues", {})
    ml = analysis.get("ml_understanding", {})
    recommendations = analysis.get("ai_recommendations", [])

    # Friendly, conversational explanation built from analysis dict
    rows = overview.get("rows", 0)
    cols = overview.get("columns", 0)
    memory = overview.get("memory_mb", 0)

    numerical = columns.get("numerical") or []
    categorical = columns.get("categorical") or []
    date_columns = columns.get("date") or []

    missing_total = quality.get("missing_values", {}).get("total", 0)
    missing_by_col = quality.get("missing_values", {}).get("by_column", {})
    duplicates = quality.get("duplicate_rows", 0)

    possible_targets = ml.get("possible_targets") or []
    suggested_target = ml.get("suggested_target")
    problem_type = ml.get("problem_type", "unknown")

    # Build short human-friendly descriptions
    if possible_targets:
        target_text = ", ".join(html.escape(str(c)) for c in possible_targets)
    elif suggested_target:
        target_text = html.escape(str(suggested_target))
    else:
        target_text = "No obvious target column detected."

    if problem_type == "classification":
        problem_text = "classification (predicting which category an item belongs to)"
    elif problem_type == "regression":
        problem_text = "regression (predicting a continuous number like price or amount)"
    else:
        problem_text = "unclear until a target column is chosen"

    # Examples of important columns: list up to 6 names for readability
    important_cols = overview.get("column_names", [])[:6]
    important_example = (
        html.escape(", ".join(map(str, important_cols)))
        if important_cols
        else "No columns available to preview"
    )

    # Missing values summary
    if missing_total > 0:
        top_missing = sorted(missing_by_col.items(), key=lambda it: it[1], reverse=True)[:3]
        missing_examples = ", ".join(f"{html.escape(str(c))} ({n})" for c, n in top_missing)
        missing_text = f"There are {missing_total:,} missing values. Top examples: {missing_examples}."
    else:
        missing_text = "I couldn't find missing values — looks complete." 

    duplicate_text = (
        f"I found {duplicates:,} duplicate rows." if duplicates else "No duplicate rows detected."
    )

    # Data types quick summary
    dtypes = columns.get("data_types", {})
    dtype_examples = []
    for k, v in list(dtypes.items())[:4]:
        dtype_examples.append(f"{html.escape(str(k))}: {html.escape(str(v))}")
    dtype_text = ", ".join(dtype_examples) if dtype_examples else "Types not available."

    # Why this is useful in plain language
    why_text = (
        "Solving this kind of problem can help automate decisions, find patterns, "
        "or predict outcomes that save time and reduce mistakes in real-world tasks."
    )

    # Next steps inside AutoDS (keep consistent with existing guidance)
    next_step = analysis.get("next_step") or NEXT_STEP

    # Compose a single conversational HTML message
    conversation = f"""
<p>Hello — I reviewed your dataset and I'll explain it like a friendly data mentor.</p>

<p><strong>1) What is this dataset about?</strong><br>
This dataset contains {rows:,} rows (think: {rows:,} separate examples or records) and {cols} columns (each column is a piece of information collected for every example). It uses about {memory} MB of memory when loaded.</p>

<p><strong>2) What does each row represent?</strong><br>
Each row is one instance — for example, one customer, one house listing, or one medical record depending on the dataset. Treat a row as a single "case" with several attributes.</p>

<p><strong>3) Important columns / features</strong><br>
Here are a few example column names: {important_example}. In general, <em>numerical</em> columns are numbers you can do math with (e.g., age, price), <em>categorical</em> columns are labels or groups (e.g., country, product type), and <em>date-like</em> columns hold timestamps or dates.</p>

<p><strong>4) Size in simple words</strong><br>
This file has {rows:,} examples and {cols} pieces of information per example — that means it's a small/medium dataset you can experiment with interactively.</p>

<p><strong>5) Data quality quick check</strong><br>
{missing_text} {duplicate_text} Data types (sample): {html.escape(dtype_text)}</p>

<p><strong>6) What machine learning problem can this solve?</strong><br>
Based on the data I found, the likely problem type is <strong>{html.escape(problem_text)}</strong>. Possible target columns I detected: {target_text}.</p>

<p><strong>7) Why this is useful in real life</strong><br>
{why_text} For instance, classification can help automate approvals, regression can estimate prices, and clustering can help find natural groups in your data.</p>

<p><strong>8) What you should do next inside AutoDS</strong><br>
- Open <strong>Data Quality Lab</strong> to handle missing values and duplicates.  
- Run <strong>EDA Explorer</strong> (click Run EDA Analysis) to inspect distributions and relationships.  
- If you have a target column, go to <strong>AutoML Studio</strong> to train models and see which works best.  
These steps help ensure models are trained on clean, understandable data.</p>

<p>If you'd like, I can walk through the top issues or show how to fix missing values now — tell me which area you want to focus on.</p>
"""

    return conversation
