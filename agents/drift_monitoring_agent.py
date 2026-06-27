"""Data drift monitoring agent for AutoDS Agent.

This backend-only module monitors incoming production data against training
reference data, detects drift and data quality issues, generates reports,
and stores drift history for lifecycle tracking.
"""
from __future__ import annotations

import datetime
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from agents.experiment_memory_agent import ExperimentMemoryAgent


class DataDriftMonitoringAgent:
    DEFAULT_REFERENCE_FILE = "storage/monitoring/drift_reference.json"
    DEFAULT_HISTORY_FILE = "drift_history.json"
    MAX_HISTORY = 1000
    NUMERIC_BINS = 10

    def __init__(
        self,
        reference_path: Optional[str] = None,
        history_path: Optional[str] = None,
    ) -> None:
        self.reference_path = reference_path or self.DEFAULT_REFERENCE_FILE
        self.history_path = history_path or self.DEFAULT_HISTORY_FILE
        self.memory_agent = ExperimentMemoryAgent()
        self.history = self._load_history()

    def _ensure_path(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _load_history(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.history_path):
            return []

        try:
            with open(self.history_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return []

    def _persist_history(self) -> None:
        self._ensure_path(self.history_path)
        with open(self.history_path, "w", encoding="utf-8") as handle:
            json.dump(self.history[-self.MAX_HISTORY :], handle, indent=2)

    def register_reference_data(
        self,
        reference_df: pd.DataFrame,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register training data statistics for drift monitoring."""
        stats = self._compute_reference_stats(reference_df)
        reference_data = {
            "dataset_name": dataset_name or "reference_dataset",
            "registered_at": datetime.datetime.utcnow().isoformat() + "Z",
            "rows": int(reference_df.shape[0]),
            "columns": int(reference_df.shape[1]),
            "feature_names": list(reference_df.columns),
            "dtypes": {col: str(dtype) for col, dtype in reference_df.dtypes.items()},
            "statistics": stats,
        }
        self._ensure_path(self.reference_path)
        with open(self.reference_path, "w", encoding="utf-8") as handle:
            json.dump(reference_data, handle, indent=2)
        return reference_data

    def _compute_reference_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        numerical = {}
        categorical = {}

        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                values = df[column].dropna().astype(float)
                if values.empty:
                    numerical[column] = {
                        "mean": None,
                        "median": None,
                        "std": None,
                        "min": None,
                        "max": None,
                        "bins": [],
                        "bin_edges": [],
                    }
                    continue

                hist, edges = np.histogram(values, bins=self.NUMERIC_BINS)
                missing_ratio = float(df[column].isna().mean())
                numerical[column] = {
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": float(values.std(ddof=0)),
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "bins": hist.tolist(),
                    "bin_edges": edges.tolist(),
                    "missing_ratio": missing_ratio,
                }
            else:
                frequencies = df[column].fillna("<MISSING>").astype(str).value_counts().to_dict()
                missing_ratio = float(df[column].isna().mean())
                categorical[column] = {
                    "frequencies": {str(k): int(v) for k, v in frequencies.items()},
                    "missing_ratio": missing_ratio,
                }

        return {
            "numerical": numerical,
            "categorical": categorical,
        }

    def load_reference_data(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self.reference_path):
            return None

        try:
            with open(self.reference_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None

    def detect_drift(
        self,
        reference_data: Dict[str, Any],
        new_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Detect drift between reference and new data."""
        numerical_results = {}
        categorical_results = {}
        feature_scores: List[float] = []

        ref_stats = reference_data.get("statistics", {})

        for feature, stats in ref_stats.get("numerical", {}).items():
            if feature not in new_data.columns:
                numerical_results[feature] = {
                    "status": "missing",
                    "psi": None,
                    "mean_shift": None,
                    "distribution_change": None,
                }
                continue

            new_values = new_data[feature].dropna().astype(float)
            if new_values.empty:
                numerical_results[feature] = {
                    "status": "no_data",
                    "psi": None,
                    "mean_shift": None,
                    "distribution_change": None,
                }
                continue

            psi = self._calculate_psi(stats, new_values)
            mean_shift = self._calculate_mean_shift(stats, new_values)
            distribution_change = self._calculate_distribution_change(stats, new_values)
            numerical_results[feature] = {
                "status": "present",
                "psi": psi,
                "mean_shift_percent": mean_shift,
                "distribution_change": distribution_change,
                "drift_level": self._drift_level(psi),
            }
            feature_scores.append(psi)

        for feature, stats in ref_stats.get("categorical", {}).items():
            if feature not in new_data.columns:
                categorical_results[feature] = {
                    "status": "missing",
                    "frequency_change": None,
                    "new_categories": [],
                }
                continue

            new_values = new_data[feature].fillna("<MISSING>").astype(str)
            frequency_change, new_categories = self._calculate_categorical_drift(stats, new_values)
            severity = self._categorical_severity(frequency_change, new_categories)
            categorical_results[feature] = {
                "status": "present",
                "frequency_change": frequency_change,
                "new_categories": new_categories,
                "drift_level": severity,
            }
            feature_scores.append(frequency_change)

        overall_score = float(np.mean(feature_scores)) if feature_scores else 0.0
        severity = self._drift_level(overall_score)

        return {
            "numerical": numerical_results,
            "categorical": categorical_results,
            "overall_drift_score": overall_score,
            "severity": severity,
            "feature_count": len(feature_scores),
        }

    def _calculate_psi(self, stats: Dict[str, Any], new_values: pd.Series) -> float:
        if not stats.get("bins") or not stats.get("bin_edges"):
            return 0.0

        ref_bins = np.array(stats["bins"], dtype=float)
        ref_total = ref_bins.sum()
        ref_pct = np.where(ref_total > 0, ref_bins / ref_total, np.zeros_like(ref_bins))

        edges = np.array(stats["bin_edges"], dtype=float)
        new_counts, _ = np.histogram(new_values, bins=edges)
        new_total = new_counts.sum()
        new_pct = np.where(new_total > 0, new_counts / new_total, np.zeros_like(new_counts))

        psi = 0.0
        for r, n in zip(ref_pct, new_pct):
            r = max(r, 1e-6)
            n = max(n, 1e-6)
            psi += (r - n) * math.log(r / n)
        return float(psi)

    def _calculate_mean_shift(self, stats: Dict[str, Any], new_values: pd.Series) -> float:
        ref_mean = stats.get("mean")
        if ref_mean is None:
            return 0.0
        new_mean = float(new_values.mean())
        denominator = abs(ref_mean) if abs(ref_mean) > 1e-6 else 1.0
        return float(abs(new_mean - ref_mean) / denominator * 100.0)

    def _calculate_distribution_change(self, stats: Dict[str, Any], new_values: pd.Series) -> float:
        if not stats.get("bins") or not stats.get("bin_edges"):
            return 0.0
        ref_bins = np.array(stats["bins"], dtype=float)
        ref_pct = ref_bins / max(ref_bins.sum(), 1)
        edges = np.array(stats["bin_edges"], dtype=float)
        new_counts, _ = np.histogram(new_values, bins=edges)
        new_pct = new_counts / max(new_counts.sum(), 1)
        return float(np.sum(np.abs(ref_pct - new_pct)) / 2.0)

    def _calculate_categorical_drift(
        self,
        stats: Dict[str, Any],
        new_values: pd.Series,
    ) -> Tuple[float, List[str]]:
        ref_freq = {str(k): int(v) for k, v in stats.get("frequencies", {}).items()}
        new_counts = new_values.value_counts().to_dict()
        total_ref = sum(ref_freq.values()) if ref_freq else 1
        total_new = sum(new_counts.values()) if new_counts else 1

        categories = set(ref_freq.keys()) | set(new_counts.keys())
        change = 0.0
        unseen = []
        for category in categories:
            ref_pct = ref_freq.get(category, 0) / total_ref
            new_pct = new_counts.get(category, 0) / total_new
            change += abs(ref_pct - new_pct)
            if category not in ref_freq:
                unseen.append(category)

        return float(change / 2.0), unseen

    def _categorical_severity(self, frequency_change: float, new_categories: List[str]) -> str:
        if new_categories:
            if frequency_change >= 0.25:
                return "High"
            return "Medium"
        if frequency_change >= 0.25:
            return "High"
        if frequency_change >= 0.10:
            return "Medium"
        return "Low"

    def _drift_level(self, score: float) -> str:
        if score >= 0.25:
            return "High"
        if score >= 0.10:
            return "Medium"
        return "Low"

    def monitor_data_quality(
        self,
        reference_data: Dict[str, Any],
        new_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Detect data quality issues and compute a health score."""
        issues: List[str] = []
        score = 100

        ref_cols = set(reference_data.get("feature_names", []))
        new_cols = set(new_data.columns.tolist())
        missing_cols = sorted(list(ref_cols - new_cols))
        new_columns = sorted(list(new_cols - ref_cols))

        if missing_cols:
            issues.append(f"Missing columns: {', '.join(missing_cols)}")
            score -= min(30, 10 * len(missing_cols))

        if new_columns:
            issues.append(f"New columns detected: {', '.join(new_columns)}")
            score -= min(20, 5 * len(new_columns))

        ref_dtypes = reference_data.get("dtypes", {})
        dtype_changes = []
        for feature in ref_cols & new_cols:
            ref_dtype = ref_dtypes.get(feature)
            new_dtype = str(new_data[feature].dtype)
            if ref_dtype != new_dtype:
                dtype_changes.append(feature)
        if dtype_changes:
            issues.append(f"Data type changes: {', '.join(dtype_changes)}")
            score -= min(20, 5 * len(dtype_changes))

        ref_stats = reference_data.get("statistics", {})
        missing_ratio_increases = []
        for feature in ref_cols & new_cols:
            ref_missing = 0.0
            if feature in ref_stats.get("numerical", {}):
                ref_missing = float(ref_stats["numerical"][feature].get("missing_ratio", 0.0))
            elif feature in ref_stats.get("categorical", {}):
                ref_missing = float(ref_stats["categorical"][feature].get("missing_ratio", 0.0))
            new_missing = float(new_data[feature].isna().mean())
            if new_missing > ref_missing + 0.01:
                missing_ratio_increases.append((feature, ref_missing, new_missing))
        if missing_ratio_increases:
            issue_features = ", ".join([feature for feature, _, _ in missing_ratio_increases])
            issues.append(f"Missing values increased for: {issue_features}.")
            score -= min(20, 5 * len(missing_ratio_increases))

        outlier_warnings = self._find_extreme_outliers(reference_data, new_data)
        if outlier_warnings:
            issues.extend(outlier_warnings)
            score -= min(20, 5 * len(outlier_warnings))

        score = max(0, min(100, score))
        return {
            "health_score": score,
            "issues": issues,
            "missing_columns": missing_cols,
            "new_columns": new_columns,
            "dtype_changes": dtype_changes,
            "outlier_warnings": outlier_warnings,
        }

    def _find_extreme_outliers(self, reference_data: Dict[str, Any], new_data: pd.DataFrame) -> List[str]:
        warnings: List[str] = []
        for feature, stats in reference_data.get("statistics", {}).get("numerical", {}).items():
            if feature not in new_data.columns:
                continue
            ref_min = stats.get("min")
            ref_max = stats.get("max")
            if ref_min is None or ref_max is None:
                continue
            values = new_data[feature].dropna().astype(float)
            if values.empty:
                continue
            if values.min() < ref_min or values.max() > ref_max:
                warnings.append(f"Extreme values detected in {feature}.")
        return warnings

    def generate_drift_report(
        self,
        reference_data: Dict[str, Any],
        new_data: pd.DataFrame,
        deployment_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a structured drift report with recommended actions."""
        drift = self.detect_drift(reference_data, new_data)
        data_quality = self.monitor_data_quality(reference_data, new_data)
        recommendation = self.recommend_action(deployment_metadata, drift)

        affected_features = [
            feature
            for feature, detail in drift.get("numerical", {}).items()
            if detail.get("drift_level") in {"Medium", "High"}
        ]
        affected_features += [
            feature
            for feature, detail in drift.get("categorical", {}).items()
            if detail.get("drift_level") in {"Medium", "High"}
        ]

        business_explanation = (
            f"Detected {drift['severity']} drift in {len(affected_features)} feature(s). "
            "Investigate model performance and data consistency before continuing." 
        )
        if drift["severity"] == "High":
            business_explanation = (
                f"Critical data drift detected in {len(affected_features)} feature(s). "
                "Retraining is recommended immediately."
            )

        report = {
            "overall_drift_status": f"{drift['severity']} Drift",
            "drift_score": drift["overall_drift_score"],
            "severity": drift["severity"],
            "features_affected": affected_features,
            "business_explanation": business_explanation,
            "recommended_action": recommendation.get("action"),
            "recommendation_reason": recommendation.get("reason"),
            "data_quality": data_quality,
            "drift_details": drift,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        self._append_history(reference_data, report)
        return report

    def recommend_action(
        self,
        deployment_metadata: Optional[Dict[str, Any]] = None,
        drift_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Recommend retraining or monitoring actions based on drift severity."""
        severity = "Low"
        if drift_report is not None:
            severity = drift_report.get("severity", "Low")

        action = "Continue monitoring."
        reason = "Drift is below the retraining threshold."
        if severity == "Medium":
            action = "Review model performance and validate incoming data."
            reason = "Moderate drift may impact predictions and should be inspected."
        elif severity == "High":
            action = "Retrain model immediately."
            reason = "High drift indicates production inputs no longer match training data."

        deployment_name = None
        if deployment_metadata:
            deployment_name = deployment_metadata.get("model_name") or deployment_metadata.get("model")

        if deployment_name:
            reason = f"Model {deployment_name}: {reason}"

        history_summary = self.memory_agent.get_best_experiment().get("best_experiment", {})
        if history_summary:
            reason += f" Last best experiment used {history_summary.get('algorithm_name', 'a model')} with score {history_summary.get('test_score')} ."

        return {
            "action": action,
            "reason": reason,
        }

    def _append_history(self, reference_data: Dict[str, Any], report: Dict[str, Any]) -> None:
        record = {
            "timestamp": report["timestamp"],
            "dataset_name": reference_data.get("dataset_name", "reference_dataset"),
            "rows": reference_data.get("rows"),
            "columns": reference_data.get("columns"),
            "drift_score": report["drift_score"],
            "severity": report["severity"],
            "recommended_action": report["recommended_action"],
            "features_affected": report["features_affected"],
        }
        self.history.append(record)
        self.history = self.history[-self.MAX_HISTORY :]
        self._persist_history()

    def generate_alert(self, drift_report: Dict[str, Any]) -> Dict[str, Any]:
        severity = drift_report.get("severity", "Low")
        count = len(drift_report.get("features_affected", []))
        if severity == "High":
            return {
                "level": "HIGH ALERT",
                "message": (
                    f"Critical data drift detected in {count} feature(s). "
                    "Model retraining recommended."
                ),
            }
        if severity == "Medium":
            return {
                "level": "WARNING",
                "message": (
                    f"Moderate drift detected in {count} feature(s). "
                    "Review required."
                ),
            }
        return {
            "level": "INFO",
            "message": "Low drift detected. Continue monitoring.",
        }
