"""Resolve and reuse on-disk chart paths for PDF embedding (no EDA re-analysis)."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.safe_checks import coalesce_dict, feature_importance_as_dict

_logger = logging.getLogger(__name__)

CHARTS_DIR = os.path.join("reports", "charts")

EDA_CHART_ORDER: Tuple[Tuple[str, str], ...] = (
    ("missing_values", "Missing Values"),
    ("correlation_heatmap", "Correlation Heatmap"),
    ("target_distribution", "Target / Class Distribution"),
    ("feature_distribution", "Feature Distribution"),
    ("outlier_boxplot", "Outlier Detection"),
    ("scatter_plot", "Feature Scatter Plot"),
)


def chart_prefix_from_name(name: Optional[str]) -> str:
    safe = re.sub(r"[^\w\-]", "_", str(name or "report")).strip("_")
    return (safe or "report")[:40]


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def _resolve_paths_from_eda_results(eda_results: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not isinstance(eda_results, dict):
        return {}
    charts = coalesce_dict(eda_results.get("charts"))
    resolved: Dict[str, str] = {}
    for key, _title in EDA_CHART_ORDER:
        val = charts.get(key)
        if isinstance(val, str) and os.path.isfile(val):
            resolved[key] = val
    return resolved


def _cached_paths(prefix: str) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    for key, _title in EDA_CHART_ORDER:
        path = os.path.join(CHARTS_DIR, f"{prefix}_{key}.png")
        if os.path.isfile(path):
            paths[key] = path
    return paths


def _export_eda_charts_matplotlib(
    df: pd.DataFrame,
    target_column: Optional[str],
    prefix: str,
) -> Dict[str, str]:
    """Write EDA PNGs using the same views as the dashboard (matplotlib export only)."""
    os.makedirs(CHARTS_DIR, exist_ok=True)
    paths: Dict[str, str] = {}
    numerics = _numeric_columns(df)

    try:
        missing = df.isnull().sum()
        plot_missing = missing[missing > 0] if (missing > 0).any() else missing
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_missing.plot(kind="bar", ax=ax, color="#EF4444")
        ax.set_title("Missing Values Analysis")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = os.path.join(CHARTS_DIR, f"{prefix}_missing_values.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths["missing_values"] = path
    except Exception as exc:
        _logger.debug("Missing values chart export skipped: %s", exc)
        plt.close("all")

    if len(numerics) >= 2:
        try:
            import seaborn as sns

            subset = numerics[:12]
            corr = df[subset].corr()
            fig, ax = plt.subplots(figsize=(7.5, 5.5))
            sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax)
            ax.set_title("Correlation Heatmap")
            plt.tight_layout()
            path = os.path.join(CHARTS_DIR, f"{prefix}_correlation_heatmap.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            paths["correlation_heatmap"] = path
        except Exception as exc:
            _logger.debug("Correlation heatmap export skipped: %s", exc)
            plt.close("all")

    target_col = target_column if target_column in df.columns else None
    if not target_col:
        categorical = [c for c in df.columns if c not in numerics]
        target_col = categorical[0] if categorical else (numerics[-1] if numerics else None)

    if target_col and target_col in df.columns:
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            series = df[target_col].dropna()
            if pd.api.types.is_numeric_dtype(series):
                ax.hist(series, bins=min(30, max(5, series.nunique())), color="#6366F1")
            else:
                series.astype(str).value_counts().head(15).iloc[::-1].plot(
                    kind="barh", ax=ax, color="#6366F1"
                )
            ax.set_title(f"Target / Class Distribution — {target_col}")
            plt.tight_layout()
            path = os.path.join(CHARTS_DIR, f"{prefix}_target_distribution.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            paths["target_distribution"] = path
        except Exception as exc:
            _logger.debug("Target distribution export skipped: %s", exc)
            plt.close("all")

    feature_col = next((c for c in numerics if c != target_col), numerics[0] if numerics else None)
    if feature_col:
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(df[feature_col].dropna(), bins=30, color="#8B5CF6")
            ax.set_title(f"Feature Distribution — {feature_col}")
            plt.tight_layout()
            path = os.path.join(CHARTS_DIR, f"{prefix}_feature_distribution.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            paths["feature_distribution"] = path
        except Exception as exc:
            _logger.debug("Feature distribution export skipped: %s", exc)
            plt.close("all")

        try:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.boxplot(df[feature_col].dropna(), orientation="vertical")
            ax.set_title(f"Outlier Detection — {feature_col}")
            ax.set_xticklabels([feature_col])
            plt.tight_layout()
            path = os.path.join(CHARTS_DIR, f"{prefix}_outlier_boxplot.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            paths["outlier_boxplot"] = path
        except Exception as exc:
            _logger.debug("Outlier boxplot export skipped: %s", exc)
            plt.close("all")

    if len(numerics) >= 2:
        x_col, y_col = numerics[0], numerics[1]
        try:
            sample_df = df[[x_col, y_col]].dropna()
            if len(sample_df) > 500:
                sample_df = sample_df.sample(500, random_state=42)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(sample_df[x_col], sample_df[y_col], alpha=0.5, color="#06B6D4", s=12)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Scatter Plot — {x_col} vs {y_col}")
            plt.tight_layout()
            path = os.path.join(CHARTS_DIR, f"{prefix}_scatter_plot.png")
            fig.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            paths["scatter_plot"] = path
        except Exception as exc:
            _logger.debug("Scatter plot export skipped: %s", exc)
            plt.close("all")

    return paths


def ensure_eda_chart_paths(
    cleaned_df: Optional[pd.DataFrame],
    target_column: Optional[str],
    eda_results: Optional[Dict[str, Any]] = None,
    chart_prefix: str = "report",
    existing_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return EDA chart paths, preferring pipeline-provided or cached files."""
    if existing_paths:
        valid = {k: v for k, v in existing_paths.items() if isinstance(v, str) and os.path.isfile(v)}
        if valid:
            return valid

    from_pipeline = _resolve_paths_from_eda_results(eda_results)
    if from_pipeline:
        return from_pipeline

    cached = _cached_paths(chart_prefix)
    if cached:
        return cached

    if cleaned_df is None or not isinstance(cleaned_df, pd.DataFrame) or cleaned_df.empty:
        return {}

    try:
        return _export_eda_charts_matplotlib(cleaned_df, target_column, chart_prefix)
    except Exception as exc:
        _logger.warning("EDA chart export failed: %s", exc)
        return {}


def ensure_explainability_chart_paths(
    explainability_results: Optional[Dict[str, Any]],
    chart_prefix: str = "report",
    existing_paths: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Return explainability chart paths, reusing pipeline or cached PNGs when possible."""
    if existing_paths:
        valid = {k: v for k, v in existing_paths.items() if isinstance(v, str) and os.path.isfile(v)}
        if valid:
            return valid

    if isinstance(explainability_results, dict):
        for key in ("chart_path", "plot_path", "feature_importance_plot", "shap_summary_plot"):
            val = explainability_results.get(key)
            if isinstance(val, str) and os.path.isfile(val):
                return {"feature_importance": val}

        charts = coalesce_dict(explainability_results.get("charts"))
        for val in charts.values():
            if isinstance(val, str) and os.path.isfile(val):
                return {"feature_importance": val}

    cached = os.path.join(CHARTS_DIR, f"{chart_prefix}_feature_importance.png")
    if os.path.isfile(cached):
        return {"feature_importance": cached}

    fi_dict = feature_importance_as_dict(
        explainability_results.get("feature_importance") if isinstance(explainability_results, dict) else None
    )
    if not fi_dict:
        return {}

    try:
        os.makedirs(CHARTS_DIR, exist_ok=True)
        ranked = sorted(fi_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:15]
        names = [str(name) for name, _ in ranked][::-1]
        values = [float(val) for _, val in ranked][::-1]
        fig, ax = plt.subplots(figsize=(7, max(4.0, len(names) * 0.28)))
        ax.barh(names, values, color="#4338CA")
        ax.set_title("Feature Importance")
        plt.tight_layout()
        fig.savefig(cached, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return {"feature_importance": cached}
    except Exception as exc:
        _logger.warning("Explainability chart export failed: %s", exc)
        plt.close("all")
        return {}


def ordered_eda_charts(paths: Dict[str, str]) -> List[Tuple[str, str]]:
    """Return (title, path) pairs in the standard EDA visualization order."""
    ordered: List[Tuple[str, str]] = []
    for key, title in EDA_CHART_ORDER:
        path = paths.get(key)
        if isinstance(path, str) and os.path.isfile(path):
            ordered.append((title, path))
    return ordered
