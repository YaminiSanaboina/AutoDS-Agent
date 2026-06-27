import pandas as pd
import numpy as np


def _letter_grade(score: float) -> str:
    if score >= 97:
        return "A+"
    if score >= 93:
        return "A"
    if score >= 90:
        return "A-"
    if score >= 87:
        return "B+"
    if score >= 83:
        return "B"
    if score >= 80:
        return "B-"
    if score >= 77:
        return "C+"
    if score >= 73:
        return "C"
    if score >= 70:
        return "C-"
    if score >= 60:
        return "D"
    return "F"


def _outlier_ratio(df: pd.DataFrame) -> float:
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        return 0.0
    outlier_flags = 0
    total = 0
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_flags += int(((series < lower) | (series > upper)).sum())
        total += len(series)
    return (outlier_flags / max(total, 1)) * 100


def _class_imbalance_ratio(df: pd.DataFrame, target_col: str | None = None) -> float | None:
    if not target_col or target_col not in df.columns:
        object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not object_cols:
            return None
        target_col = object_cols[-1]
    counts = df[target_col].value_counts(normalize=True, dropna=True)
    if len(counts) < 2:
        return None
    return float(counts.iloc[0] * 100)


def compute_health_score(df, target_column: str | None = None):
    if df is None or df.empty:
        return {
            "score": 0,
            "grade": "N/A",
            "letter_grade": "N/A",
            "grade_class": "health-poor",
            "breakdown": {},
            "summary": "No dataset loaded.",
            "recommendations": [],
        }

    rows, cols = df.shape
    missing_pct = df.isnull().sum().sum() / max(rows * cols, 1) * 100
    duplicate_pct = df.duplicated().sum() / max(rows, 1) * 100
    outlier_pct = _outlier_ratio(df)
    constant_cols = sum(df[col].nunique(dropna=False) <= 1 for col in df.columns)
    constant_pct = constant_cols / max(cols, 1) * 100
    imbalance_pct = _class_imbalance_ratio(df, target_column)

    missing_score = max(0, 100 - missing_pct * 2)
    duplicate_score = max(0, 100 - duplicate_pct * 5)
    diversity_score = max(0, 100 - constant_pct * 3)
    completeness_score = max(0, 100 - (df.isnull().any(axis=1).sum() / max(rows, 1) * 100))
    outlier_score = max(0, 100 - outlier_pct * 1.5)
    consistency_score = max(0, 100 - constant_pct * 4)
    imbalance_penalty = 0.0
    if imbalance_pct is not None and imbalance_pct > 85:
        imbalance_penalty = min(15, (imbalance_pct - 85) * 0.5)

    overall = (
        missing_score * 0.25
        + duplicate_score * 0.15
        + diversity_score * 0.15
        + completeness_score * 0.15
        + outlier_score * 0.15
        + consistency_score * 0.15
    ) - imbalance_penalty
    overall = round(min(100, max(0, overall)), 1)

    letter = _letter_grade(overall)
    if overall >= 85:
        grade, grade_class = "Excellent", "health-excellent"
    elif overall >= 70:
        grade, grade_class = "Good", "health-good"
    elif overall >= 50:
        grade, grade_class = "Fair", "health-fair"
    else:
        grade, grade_class = "Poor", "health-poor"

    recommendations = _build_recommendations(
        df, missing_pct, duplicate_pct, outlier_pct, constant_cols, imbalance_pct, target_column
    )

    return {
        "score": overall,
        "grade": grade,
        "letter_grade": letter,
        "grade_class": grade_class,
        "breakdown": {
            "Missing Values": round(missing_score, 1),
            "Duplicates": round(duplicate_score, 1),
            "Outliers": round(outlier_score, 1),
            "Feature Diversity": round(diversity_score, 1),
            "Row Completeness": round(completeness_score, 1),
            "Data Consistency": round(consistency_score, 1),
        },
        "summary": _health_summary(overall, missing_pct, duplicate_pct, constant_cols, outlier_pct),
        "recommendations": recommendations,
        "missing_pct": round(missing_pct, 2),
        "duplicate_pct": round(duplicate_pct, 2),
        "outlier_pct": round(outlier_pct, 2),
        "imbalance_pct": round(imbalance_pct, 2) if imbalance_pct is not None else None,
    }


def _build_recommendations(df, missing_pct, duplicate_pct, outlier_pct, constant_cols, imbalance_pct, target_column):
    recs = []
    if missing_pct > 5:
        worst = df.isnull().sum().sort_values(ascending=False)
        worst_col = worst.index[0] if len(worst) else None
        if worst_col and worst[worst_col] > 0:
            recs.append(f"Impute or remove missing values in '{worst_col}' ({int(worst[worst_col])} gaps).")
    if duplicate_pct > 0:
        recs.append(f"Remove {int(df.duplicated().sum())} duplicate rows to avoid biased training.")
    if outlier_pct > 5:
        recs.append("Review numeric outliers using boxplots — cap or transform extreme values.")
    if constant_cols > 0:
        recs.append("Drop constant columns that provide no predictive signal.")
    if imbalance_pct is not None and imbalance_pct > 85:
        recs.append("Apply SMOTE or class weighting to address severe class imbalance.")
    if not recs:
        recs.append("Dataset quality is strong — proceed to Run Analysis.")
    return recs


def _health_summary(score, missing_pct, duplicate_pct, constant_cols, outlier_pct):
    parts = []
    if score >= 85:
        parts.append("Dataset quality is strong and ready for modeling.")
    elif score >= 70:
        parts.append("Dataset is in good shape with minor improvements possible.")
    elif score >= 50:
        parts.append("Dataset has notable quality issues that should be addressed.")
    else:
        parts.append("Dataset requires significant cleaning before reliable modeling.")

    if missing_pct > 5:
        parts.append(f"Missing values affect {missing_pct:.1f}% of cells.")
    if duplicate_pct > 0:
        parts.append(f"Duplicate rows represent {duplicate_pct:.1f}% of records.")
    if outlier_pct > 5:
        parts.append(f"Outliers detected in approximately {outlier_pct:.1f}% of numeric values.")
    if constant_cols > 0:
        parts.append(f"{constant_cols} constant or near-constant column(s) detected.")

    return " ".join(parts)


def detect_data_issues(df):
    issues = []

    if df is None or df.empty:
        return issues

    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append({
            "severity": "medium",
            "title": "Duplicate Rows",
            "description": f"{duplicates} duplicate row(s) found.",
            "recommendation": "Remove duplicates to prevent biased model training.",
        })

    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing == 0:
            continue

        pct = missing / len(df) * 100
        severity = "critical" if pct > 30 else "medium" if pct > 10 else "low"

        fill_strategy = (
            "median imputation"
            if pd.api.types.is_numeric_dtype(df[col])
            else "mode imputation"
        )

        issues.append({
            "severity": severity,
            "title": f"Missing Values in '{col}'",
            "description": f"{missing} missing values ({pct:.1f}% of rows).",
            "recommendation": f"Apply {fill_strategy} or investigate data collection gaps.",
        })

    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            issues.append({
                "severity": "medium",
                "title": f"Constant Feature '{col}'",
                "description": "Column has a single unique value.",
                "recommendation": "Drop this column — it provides no predictive signal.",
            })

    high_cardinality = [
        col for col in df.select_dtypes(include="object").columns
        if df[col].nunique() > 50
    ]
    for col in high_cardinality:
        issues.append({
            "severity": "low",
            "title": f"High Cardinality in '{col}'",
            "description": f"{df[col].nunique()} unique categories detected.",
            "recommendation": "Consider grouping rare categories or target encoding.",
        })

    if not issues:
        issues.append({
            "severity": "low",
            "title": "No Critical Issues",
            "description": "Dataset passed automated quality checks.",
            "recommendation": "Proceed to EDA and modeling.",
        })

    return issues
