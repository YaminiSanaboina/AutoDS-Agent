"""Autonomous retraining agent for AutoDS Agent.

This backend-only module decides whether retraining is needed, prepares data,
re-trains candidate models, compares against an existing model, and prepares a
new deployment package while preserving existing APIs.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    r2_score,
)
from sklearn.model_selection import train_test_split

from agents.deployment_agent import DeploymentAgent
from agents.experiment_memory_agent import ExperimentMemoryAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.hyperparameter_agent import HyperparameterOptimizationAgent
from agents.self_healing_agent import SelfHealingAgent
from agents.supervisor_agent import AISupervisor


class AutonomousRetrainingAgent:
    DEFAULT_HISTORY_FILE = "retraining_history.json"
    MAX_HISTORY = 500
    DEFAULT_IMPROVEMENT_THRESHOLD = 0.02

    def __init__(
        self,
        history_path: Optional[str] = None,
        experiment_memory_path: Optional[str] = None,
        package_dir: Optional[str] = None,
    ) -> None:
        self.history_path = history_path or self.DEFAULT_HISTORY_FILE
        self.package_dir = package_dir
        self.hyperparameter_agent = HyperparameterOptimizationAgent()
        self.memory_agent = ExperimentMemoryAgent(experiment_memory_path)
        self.deployment_agent = DeploymentAgent(package_dir=package_dir)
        self.self_healing_agent = SelfHealingAgent()
        self.supervisor = self._load_supervisor()
        self.history = self._load_history()

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
        directory = os.path.dirname(self.history_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(self.history_path, "w", encoding="utf-8") as handle:
            json.dump(self.history[-self.MAX_HISTORY :], handle, indent=2)

    def _load_supervisor(self) -> Optional[AISupervisor]:
        try:
            return AISupervisor()
        except Exception:
            return None

    def should_retrain(
        self,
        drift_report: Dict[str, Any],
        current_model_performance: Optional[Dict[str, float]] = None,
        deployment_metadata: Optional[Dict[str, Any]] = None,
        business_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD,
    ) -> Dict[str, Any]:
        severity = drift_report.get("severity", "Low")
        threshold = business_threshold
        if deployment_metadata:
            override = deployment_metadata.get("retraining_threshold")
            if isinstance(override, (int, float)):
                threshold = float(override)

        current_model_performance = current_model_performance or {}
        performance_drop = float(current_model_performance.get("performance_drop_pct", 0.0))

        if severity == "High":
            return {
                "decision": True,
                "reason": "High drift detected.",
                "priority": "Critical",
            }

        if severity == "Medium":
            if performance_drop >= threshold:
                return {
                    "decision": True,
                    "reason": (
                        f"Medium drift detected and model performance dropped {performance_drop * 100:.1f}%.",
                    ),
                    "priority": "High",
                }
            return {
                "decision": False,
                "reason": "Medium drift detected but model performance has not dropped enough to retrain.",
                "priority": "Moderate",
            }

        return {
            "decision": False,
            "reason": "Low drift detected. Continue monitoring.",
            "priority": "Low",
        }

    def prepare_training_data(
        self,
        df: pd.DataFrame,
        target_column: str,
        dataset_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' is missing from training data.")

        dataset_name = dataset_name or "retraining_dataset"
        feature_agent = FeatureEngineeringAgent(df, dataset_name)
        analysis = feature_agent.analyze_features()
        leakage = feature_agent.detect_leakage()
        encoding = feature_agent.recommend_encoding()
        missing_strategy = feature_agent.recommend_missing_value_strategy()
        feature_plan = feature_agent.generate_feature_plan()

        cleaned_df = df.copy()
        transformation_log: List[str] = []

        suspicious = [col for col in leakage.get("suspicious_columns", []) if col != target_column]
        if suspicious:
            cleaned_df = cleaned_df.drop(columns=suspicious, errors="ignore")
            transformation_log.append(f"Dropped suspicious leakage columns: {', '.join(suspicious)}")

        for col, strategy in missing_strategy.items():
            if col == target_column or col not in cleaned_df.columns:
                continue
            if "median" in strategy.lower() and pd.api.types.is_numeric_dtype(cleaned_df[col]):
                fill_value = cleaned_df[col].median()
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                transformation_log.append(f"Filled missing numeric values in {col} with median {fill_value}.")
            elif "mode" in strategy.lower():
                mode = cleaned_df[col].mode(dropna=True)
                if not mode.empty:
                    cleaned_df[col] = cleaned_df[col].fillna(mode.iloc[0])
                    transformation_log.append(f"Filled missing values in {col} with mode '{mode.iloc[0]}'.")
            elif "forward" in strategy.lower():
                cleaned_df[col] = cleaned_df[col].fillna(method="ffill").fillna(method="bfill")
                transformation_log.append(f"Imputed missing values in {col} with forward/backward fill.")

        for col, encoding_strategy in encoding.items():
            if col == target_column or col not in cleaned_df.columns:
                continue
            if "convert true/false" in encoding_strategy.lower():
                try:
                    cleaned_df[col] = cleaned_df[col].astype(int)
                    transformation_log.append(f"Converted boolean field {col} to integer codes.")
                except Exception:
                    transformation_log.append(f"Skipped boolean conversion for {col} due to unexpected values.")
            elif "one-hot" in encoding_strategy.lower():
                try:
                    cleaned_df = pd.get_dummies(cleaned_df, columns=[col], prefix=col, dummy_na=False)
                    transformation_log.append(f"One-hot encoded categorical field {col}.")
                except Exception:
                    codes, uniques = pd.factorize(cleaned_df[col].astype(str))
                    cleaned_df[col] = codes
                    transformation_log.append(f"Fallback encoded {col} with integer codes.")
            elif "target encoding" in encoding_strategy.lower() or "frequency encoding" in encoding_strategy.lower() or "label encoding" in encoding_strategy.lower():
                try:
                    cleaned_df[col], _ = pd.factorize(cleaned_df[col].astype(str))
                    transformation_log.append(f"Encoded categorical field {col} using integer factorization.")
                except Exception:
                    transformation_log.append(f"Skipped encoding for {col} due to unexpected values.")
            elif "text" in encoding_strategy.lower():
                cleaned_df[col] = cleaned_df[col].astype(str).fillna("")
                transformation_log.append(f"Preserved text field {col} for text feature extraction.")

        for col in cleaned_df.select_dtypes(include=["bool"]).columns:
            cleaned_df[col] = cleaned_df[col].astype(int)
            transformation_log.append(f"Converted boolean feature {col} to numeric.")

        object_columns = [c for c in cleaned_df.columns if cleaned_df[c].dtype == object and c != target_column]
        for col in object_columns:
            try:
                cleaned_df[col], _ = pd.factorize(cleaned_df[col].astype(str))
                transformation_log.append(f"Factorized object field {col} into numeric codes.")
            except Exception:
                transformation_log.append(f"Unable to factorize object field {col}; leaving as-is.")

        return {
            "cleaned_data": cleaned_df,
            "transformation_log": transformation_log,
            "feature_plan": feature_plan,
            "feature_analysis": analysis,
            "leakage": leakage,
        }

    def retrain_model(
        self,
        df: pd.DataFrame,
        target_column: str,
        problem_type: Optional[str] = None,
        dataset_name: Optional[str] = None,
        deployment_metadata: Optional[Dict[str, Any]] = None,
        package_path: Optional[str] = None,
        current_model: Optional[Any] = None,
        current_model_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        try:
            prep = self.prepare_training_data(df, target_column, dataset_name)
            cleaned = prep["cleaned_data"]
            y = cleaned[target_column]
            X = cleaned.drop(columns=[target_column], errors="ignore")
            if X.empty:
                raise ValueError("No feature columns remain after preprocessing.")

            problem_type = problem_type or self.hyperparameter_agent.detect_problem_type(y)
            model_names = self.hyperparameter_agent.supported_models(problem_type)
            results: List[Dict[str, Any]] = []
            failures: List[Dict[str, Any]] = []

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y if problem_type == "Classification" and y.nunique(dropna=True) > 1 else None,
            )

            for model_name in model_names:
                optimization = self.hyperparameter_agent.optimize(
                    X,
                    y,
                    model_name=model_name,
                    problem_type=problem_type,
                    use_history=False,
                )
                if optimization.get("status") != "success":
                    failures.append({"model_name": model_name, "error": optimization.get("diagnostics", "optimization failed")})
                    continue

                best_params = optimization.get("best_params", {})
                model = self.hyperparameter_agent._get_estimator(model_name, problem_type)
                model.set_params(**best_params)
                model.fit(X_train, y_train)
                metrics = self._evaluate_model(model, X_test, y_test, problem_type)
                primary_score = metrics["f1_score"] if problem_type == "Classification" else metrics["r2_score"]
                results.append(
                    {
                        "model_name": model_name,
                        "model": model,
                        "metrics": metrics,
                        "primary_score": primary_score,
                        "hyperparameters": best_params,
                        "training_time": optimization.get("training_duration", 0.0),
                        "optimization_report": optimization.get("report", {}),
                    }
                )

            if not results:
                error_message = "No retraining candidate succeeded."
                error_analysis = self.self_healing_agent.analyze_error(error_message, {"dataset_size": len(X)})
                recovery = self.self_healing_agent.recommend_fix(error_analysis)
                return {
                    "status": "failed",
                    "reason": error_message,
                    "failures": failures,
                    "error_analysis": error_analysis,
                    "recovery": recovery,
                }

            best_result = max(results, key=lambda entry: entry["primary_score"])
            new_model = best_result["model"]
            new_model.fit(X, y)

            compare_result = self.compare_models(
                current_model,
                new_model,
                X_test,
                y_test,
                problem_type,
            )

            package_dir = package_path or self.package_dir
            package = self.deployment_agent.package_model(
                model=new_model,
                model_name=f"AutonomousRetrained_{dataset_name or 'model'}",
                dataset_name=dataset_name or "retraining_dataset",
                problem_type=problem_type,
                feature_list=list(X.columns),
                metrics=best_result["metrics"],
                hyperparameters=best_result["hyperparameters"],
                training_info={
                    "feature_plan": prep["feature_plan"],
                    "retraining_reason": deployment_metadata.get("retraining_trigger") if deployment_metadata else None,
                    "drift_severity": deployment_metadata.get("drift_report", {}).get("severity") if deployment_metadata else None,
                    "training_time": best_result["training_time"],
                },
                package_path=package_dir,
            )

            experiment_log = {
                "dataset_name": dataset_name or "retraining_dataset",
                "dataset_shape": [len(df), len(df.columns)],
                "problem_type": problem_type,
                "algorithm_name": best_result["model_name"],
                "hyperparameters": best_result["hyperparameters"],
                "train_score": best_result["metrics"].get("training_score"),
                "test_score": best_result["metrics"].get("primary_score"),
                "training_time": best_result["training_time"],
                "feature_count": X.shape[1],
                "feature_engineering_steps": prep["feature_plan"].get("steps", []),
                "data_cleaning_steps": prep["transformation_log"],
                "notes": compare_result.get("reason"),
            }
            try:
                self.memory_agent.log_experiment(**experiment_log)
            except Exception:
                failure = traceback.format_exc()
                self.self_healing_agent.analyze_error(failure, {"dataset_size": len(X)})

            report = self.generate_retraining_report(
                trigger_reason=deployment_metadata.get("trigger_reason") if deployment_metadata else "Autonomous retraining",
                drift_severity=deployment_metadata.get("drift_report", {}).get("severity") if deployment_metadata else "Unknown",
                data_changes=prep["feature_analysis"],
                models_evaluated=[{"model_name": r["model_name"], "metrics": r["metrics"]} for r in results],
                best_model=best_result["model_name"],
                performance_comparison=compare_result,
                deployment_recommendation=compare_result,
                risks=[prep["leakage"], prep["feature_plan"]],
                limitations=["Model should be monitored after deployment.", "Retrain again if drift persists."],
            )

            self._append_history(
                {
                    "trigger_reason": deployment_metadata.get("trigger_reason") if deployment_metadata else "Autonomous retrain",
                    "drift_severity": deployment_metadata.get("drift_report", {}).get("severity") if deployment_metadata else "Unknown",
                    "dataset_version": dataset_name or "retraining_dataset",
                    "old_model_score": current_model_metrics or {},
                    "new_model_score": best_result["metrics"],
                    "deployment_decision": compare_result.get("deploy_new_model", False),
                }
            )

            if self.supervisor:
                try:
                    self.supervisor.add_decision_log("retraining", {"status": "completed", "best_model": best_result["model_name"]})
                except Exception:
                    pass

            return {
                "status": "success",
                "package": package,
                "model_card": model_card,
                "best_model_name": best_result["model_name"],
                "best_metrics": best_result["metrics"],
                "best_hyperparameters": best_result["hyperparameters"],
                "training_report": best_result["optimization_report"],
                "compare_result": compare_result,
                "retraining_report": report,
                "transformation_log": prep["transformation_log"],
            }
        except Exception as error:
            diagnostics = traceback.format_exc()
            error_analysis = self.self_healing_agent.analyze_error(str(error), {"dataset_size": len(df)})
            recovery = self.self_healing_agent.recommend_fix(error_analysis)
            return {
                "status": "failed",
                "diagnostics": str(error),
                "stacktrace": diagnostics,
                "error_analysis": error_analysis,
                "recovery": recovery,
            }

    def compare_models(
        self,
        old_model: Optional[Any],
        new_model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        problem_type: str,
    ) -> Dict[str, Any]:
        new_metrics = self._evaluate_model(new_model, X_test, y_test, problem_type)
        if old_model is None:
            return {
                "deploy_new_model": True,
                "improvement": 1.0,
                "reason": f"No existing model provided; new model achieved {new_metrics.get('primary_metric')}.",
                "old_metrics": None,
                "new_metrics": new_metrics,
            }

        old_metrics = self._evaluate_model(old_model, X_test, y_test, problem_type)
        metric_key = "f1_score" if problem_type == "Classification" else "r2_score"
        old_value = float(old_metrics.get(metric_key, 0.0))
        new_value = float(new_metrics.get(metric_key, 0.0))
        improvement = (new_value - old_value) / max(abs(old_value), 1e-6)
        deploy = improvement >= self.DEFAULT_IMPROVEMENT_THRESHOLD
        reason = (
            f"New model improved {metric_key.replace('_', ' ')} by {improvement * 100:.1f}%"
            if deploy
            else f"New model did not improve enough over the existing model ({metric_key})."
        )
        return {
            "deploy_new_model": deploy,
            "improvement": improvement,
            "reason": reason,
            "old_metrics": old_metrics,
            "new_metrics": new_metrics,
        }

    def _evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        problem_type: str,
    ) -> Dict[str, Any]:
        predictions = model.predict(X_test)
        evaluation: Dict[str, Any] = {}
        if problem_type == "Classification":
            evaluation["accuracy_score"] = float(accuracy_score(y_test, predictions))
            evaluation["precision_score"] = float(
                precision_score(y_test, predictions, average="weighted", zero_division=0)
            )
            evaluation["recall_score"] = float(
                recall_score(y_test, predictions, average="weighted", zero_division=0)
            )
            evaluation["f1_score"] = float(
                f1_score(y_test, predictions, average="weighted", zero_division=0)
            )
            evaluation["roc_auc_score"] = self._safe_roc_auc(model, X_test, y_test)
            evaluation["primary_metric"] = evaluation["f1_score"]
        else:
            evaluation["r2_score"] = float(r2_score(y_test, predictions))
            evaluation["mae"] = float(mean_absolute_error(y_test, predictions))
            evaluation["rmse"] = float(math.sqrt(mean_squared_error(y_test, predictions)))
            evaluation["primary_metric"] = evaluation["r2_score"]
        return evaluation

    def _safe_roc_auc(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Optional[float]:
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test)
                if proba.shape[1] == 2:
                    return float(roc_auc_score(y_test, proba[:, 1]))
            if hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                return float(roc_auc_score(y_test, scores))
        except Exception:
            return None
        return None

    def generate_retraining_report(
        self,
        trigger_reason: str,
        drift_severity: str,
        data_changes: Dict[str, Any],
        models_evaluated: List[Dict[str, Any]],
        best_model: str,
        performance_comparison: Dict[str, Any],
        deployment_recommendation: Dict[str, Any],
        risks: List[Any],
        limitations: List[str],
    ) -> Dict[str, Any]:
        return {
            "trigger_reason": trigger_reason,
            "drift_severity": drift_severity,
            "data_changes": data_changes,
            "models_evaluated": models_evaluated,
            "best_model": best_model,
            "performance_comparison": performance_comparison,
            "deployment_recommendation": deployment_recommendation,
            "risks": risks,
            "limitations": limitations,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }

    def _append_history(self, record: Dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "trigger_reason": record.get("trigger_reason"),
            "drift_severity": record.get("drift_severity"),
            "dataset_version": record.get("dataset_version"),
            "old_model_score": record.get("old_model_score"),
            "new_model_score": record.get("new_model_score"),
            "deployment_decision": record.get("deployment_decision"),
        }
        self.history.append(entry)
        self.history = self.history[-self.MAX_HISTORY :]
        self._persist_history()

    def get_retraining_history(self) -> List[Dict[str, Any]]:
        return list(self.history)
