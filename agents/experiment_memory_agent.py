"""Experiment tracking and AI memory agent for AutoDS Agent.

This backend-only module saves experiment history to disk, learns from past runs,
compares experiments, and recommends future AutoML actions.
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, List, Optional


class ExperimentMemoryAgent:
    """Agent that stores and learns from ML experiments."""

    DEFAULT_STORAGE = "storage/memory/auto_ds_memory.json"
    MAX_EXPERIMENTS = 500

    def __init__(self, storage_path: Optional[str] = None) -> None:
        self.storage_path = storage_path or os.path.join(os.getcwd(), self.DEFAULT_STORAGE)
        self.memory: Dict[str, List[Dict[str, Any]]] = {
            "experiments": [],
            "failures": [],
        }
        self._load_memory()

    def _load_memory(self) -> None:
        if not os.path.exists(self.storage_path):
            self._persist_memory()
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                raise ValueError("Memory file root must be a JSON object.")
            self.memory["experiments"] = data.get("experiments", []) or []
            self.memory["failures"] = data.get("failures", []) or []
        except (json.JSONDecodeError, ValueError, OSError):
            backup_path = f"{self.storage_path}.corrupt.{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}.bak"
            try:
                os.replace(self.storage_path, backup_path)
            except OSError:
                pass
            self.memory = {"experiments": [], "failures": []}
            self._persist_memory()

    def _persist_memory(self) -> None:
        directory = os.path.dirname(self.storage_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as handle:
            json.dump(self.memory, handle, indent=2)

    def _next_experiment_id(self) -> str:
        return f"EXP_{len(self.memory['experiments']) + 1:03d}"

    def log_experiment(
        self,
        dataset_name: str,
        dataset_shape: Optional[List[int]] = None,
        problem_type: Optional[str] = None,
        algorithm_name: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        train_score: Optional[float] = None,
        test_score: Optional[float] = None,
        cv_score: Optional[float] = None,
        training_time: Optional[float] = None,
        feature_count: Optional[int] = None,
        feature_engineering_steps: Optional[List[str]] = None,
        data_cleaning_steps: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> Dict[str, str]:
        experiment_id = self._next_experiment_id()
        entry: Dict[str, Any] = {
            "experiment_id": experiment_id,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "dataset_name": dataset_name,
            "dataset_shape": dataset_shape or [],
            "problem_type": problem_type or "Unknown",
            "algorithm_name": algorithm_name or "Unknown",
            "hyperparameters": hyperparameters or {},
            "train_score": train_score,
            "test_score": test_score,
            "cv_score": cv_score,
            "training_time": training_time,
            "feature_count": feature_count,
            "feature_engineering_steps": feature_engineering_steps or [],
            "data_cleaning_steps": data_cleaning_steps or [],
            "notes": notes or "",
        }
        self.memory["experiments"].append(entry)
        self.memory["experiments"] = self.memory["experiments"][-self.MAX_EXPERIMENTS :]
        self._persist_memory()
        return {"experiment_id": experiment_id, "status": "logged"}

    def get_history(
        self,
        dataset_name: Optional[str] = None,
        algorithm_name: Optional[str] = None,
        sort_by: str = "test_score",
        descending: bool = True,
        latest: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        experiments = list(self.memory["experiments"])
        if dataset_name:
            key = dataset_name.strip().lower()
            experiments = [exp for exp in experiments if exp.get("dataset_name", "").strip().lower() == key]
        if algorithm_name:
            key = algorithm_name.strip().lower()
            experiments = [exp for exp in experiments if exp.get("algorithm_name", "").strip().lower() == key]
        if sort_by and experiments:
            experiments.sort(key=lambda exp: exp.get(sort_by) if exp.get(sort_by) is not None else -1, reverse=descending)
        if latest is not None:
            experiments = experiments[:latest]
        return experiments

    def get_best_experiment(self) -> Dict[str, Any]:
        experiments = self.get_history(sort_by="test_score", descending=True)
        if not experiments:
            return {"best_experiment": {}, "reasons": ["No experiments logged yet."]}

        best = experiments[0]
        reasons: List[str] = []
        if best.get("test_score") is not None:
            reasons.append(f"Highest test score: {best['test_score']}")
        if best.get("cv_score") is not None:
            reasons.append(f"Cross-validation score: {best['cv_score']}")
        overfit_gap = self._overfit_gap(best)
        if overfit_gap is not None:
            if overfit_gap > 0.1:
                reasons.append("This experiment has a moderate overfitting gap.")
            else:
                reasons.append("This experiment shows stable generalization.")
        return {"best_experiment": best, "reasons": reasons}

    def compare_experiments(self, dataset_name: Optional[str] = None) -> Dict[str, Any]:
        experiments = self.get_history(dataset_name=dataset_name, sort_by="test_score", descending=True)
        comparison: Dict[str, Any] = {}
        for exp in experiments[:3]:
            name = exp.get("algorithm_name", "Unknown")
            score = exp.get("test_score")
            training_time = exp.get("training_time")
            comparison[name] = {
                "accuracy": f"{score:.2%}" if isinstance(score, (int, float)) else "Unknown",
                "strengths": self._algorithm_strengths(name),
                "weaknesses": self._algorithm_weaknesses(name, training_time),
            }
        return comparison

    def _algorithm_strengths(self, algorithm_name: str) -> List[str]:
        name = algorithm_name.lower()
        if "random forest" in name:
            return ["Handles non-linear data", "Robust to outliers"]
        if "xgboost" in name or "gradient" in name or "gb" in name:
            return ["Better predictive performance", "Strong with feature engineering"]
        if "logistic" in name:
            return ["Good interpretability", "Fast to train"]
        if "svm" in name:
            return ["Effective in high-dimensional spaces"]
        return ["Suitable for the task based on previous performance"]

    def _algorithm_weaknesses(self, algorithm_name: str, training_time: Optional[float]) -> List[str]:
        name = algorithm_name.lower()
        if "random forest" in name:
            return ["Higher memory usage", "Can be slower with many trees"]
        if "xgboost" in name or "gradient" in name or "gb" in name:
            return ["Longer training time", "Requires careful hyperparameter tuning"]
        if "logistic" in name:
            return ["May underperform on non-linear data"]
        if "svm" in name:
            return ["Can be slow for large datasets"]
        if training_time is not None and training_time > 300:
            return ["Long training time"]
        return ["No significant weaknesses identified"]

    def _overfit_gap(self, experiment: Dict[str, Any]) -> Optional[float]:
        train = experiment.get("train_score")
        test = experiment.get("test_score")
        try:
            if train is None or test is None:
                return None
            return float(train) - float(test)
        except (TypeError, ValueError):
            return None

    def learn_from_history(self) -> Dict[str, Any]:
        experiments = self.memory["experiments"]
        if not experiments:
            return {"recommendations": ["No history available yet."]}

        recommendations: List[str] = []
        best_by_dataset: Dict[str, Dict[str, Any]] = {}
        for exp in experiments:
            dataset = exp.get("dataset_name", "").lower()
            current = best_by_dataset.get(dataset)
            if current is None or self._score_experiment(exp) > self._score_experiment(current):
                best_by_dataset[dataset] = exp

        for dataset, exp in best_by_dataset.items():
            algo = exp.get("algorithm_name", "Unknown")
            if any(keyword in dataset for keyword in ["heart", "disease", "health", "clinical"]):
                if "random forest" in algo.lower() or "xgboost" in algo.lower():
                    recommendations.append("Random Forest or XGBoost performs best for healthcare datasets.")
            if exp.get("feature_engineering_steps"):
                if "gradient" in algo.lower() or "xgboost" in algo.lower() or "boost" in algo.lower():
                    recommendations.append("Gradient Boosting performs well after feature engineering.")
            if "logistic" in algo.lower():
                recommendations.append("Logistic Regression provides better interpretability for simpler datasets.")

        if not recommendations:
            recommendations.append("Review top-performing algorithms and consider similar strategies for new projects.")
        return {"recommendations": recommendations}

    def _score_experiment(self, experiment: Dict[str, Any]) -> float:
        score = experiment.get("test_score") or experiment.get("cv_score") or experiment.get("train_score") or 0.0
        try:
            return float(score)
        except (TypeError, ValueError):
            return 0.0

    def record_failure(
        self,
        dataset_name: str,
        algorithm_name: Optional[str],
        error_message: str,
        failure_reason: str,
        suggested_solution: str,
    ) -> Dict[str, str]:
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "dataset_name": dataset_name,
            "algorithm_name": algorithm_name or "Unknown",
            "error_message": error_message,
            "failure_reason": failure_reason,
            "suggested_solution": suggested_solution,
        }
        self.memory["failures"].append(entry)
        self._persist_memory()
        return {"status": "failure_recorded"}

    def recommend_next_experiment(self) -> Dict[str, Any]:
        history = self.memory["experiments"]
        failures = self.memory["failures"]
        plan: List[str] = []
        if history:
            best = self.get_best_experiment().get("best_experiment", {})
            if best:
                algo = best.get("algorithm_name", "").lower()
                gap = self._overfit_gap(best)
                if "random forest" in algo:
                    plan.append("Try increasing number of trees from 100 to 300 and apply feature selection.")
                if "xgboost" in algo or "gradient" in algo:
                    plan.append("Tune learning rate and max depth for better performance.")
                if "logistic" in algo:
                    plan.append("Add interaction terms or polynomial features for non-linear relationships.")
                if gap is not None and gap > 0.1:
                    plan.append("Reduce overfitting with regularization or cross-validation.")

        if failures:
            latest_failure = failures[-1]
            plan.append(f"Fix latest failure: {latest_failure.get('failure_reason')}. {latest_failure.get('suggested_solution')}")

        if not plan:
            plan.append("Try a strong ensemble algorithm or more advanced feature engineering.")

        return {"recommendation": plan}
