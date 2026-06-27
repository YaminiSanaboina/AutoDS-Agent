"""Hyperparameter optimization agent for AutoDS Agent.

This backend-only module provides intelligent hyperparameter search, comparison,
learning from past experiments, and robust failure recovery.
"""
from __future__ import annotations

import datetime
import math
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from skopt import BayesSearchCV  # type: ignore
    HAS_BAYES = True
except ImportError:
    HAS_BAYES = False

from agents.experiment_memory_agent import ExperimentMemoryAgent
from agents.self_healing_agent import SelfHealingAgent


class HyperparameterOptimizationAgent:
    """Agent that performs automated hyperparameter tuning."""

    MODEL_REGISTRY = {
        "Classification": {
            "Logistic Regression": LogisticRegression(max_iter=1000, solver="liblinear"),
            "Random Forest": RandomForestClassifier(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42) if HAS_XGB else None,
        },
        "Regression": {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
            "XGBoost Regressor": XGBRegressor(random_state=42) if HAS_XGB else None,
        },
    }

    PARAM_GRID = {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear"],
        },
        "Linear Regression": {
            "fit_intercept": [True, False],
        },
        "Random Forest": {
            "n_estimators": [50, 100, 200, 300, 400, 500],
            "max_depth": [3, 5, 10, 20, 30, 40, 50],
            "min_samples_split": [2, 5, 10, 15, 20],
        },
        "Random Forest Regressor": {
            "n_estimators": [50, 100, 200, 300, 400, 500],
            "max_depth": [3, 5, 10, 20, 30, 40, 50],
            "min_samples_split": [2, 5, 10, 15, 20],
        },
        "SVM": {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly"],
        },
        "XGBoost": {
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "max_depth": [3, 5, 7, 9, 12, 15],
            "subsample": [0.5, 0.7, 0.8, 0.9, 1.0],
        },
        "XGBoost Regressor": {
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "max_depth": [3, 5, 7, 9, 12, 15],
            "subsample": [0.5, 0.7, 0.8, 0.9, 1.0],
        },
        "Gradient Boosting": {
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "n_estimators": [50, 100, 200, 300, 400, 500],
        },
        "Gradient Boosting Regressor": {
            "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
            "n_estimators": [50, 100, 200, 300, 400, 500],
        },
    }

    SCORING = {
        "Classification": ["accuracy", "f1", "roc_auc"],
        "Regression": ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
    }

    def __init__(self, memory_path: Optional[str] = None) -> None:
        self.memory_agent = ExperimentMemoryAgent(memory_path)
        self.self_healing_agent = SelfHealingAgent()

    def supported_models(self, problem_type: str) -> List[str]:
        return [name for name, model in self.MODEL_REGISTRY.get(problem_type, {}).items() if model is not None]

    def detect_problem_type(self, y: pd.Series) -> str:
        if y.dtype.kind in "biufc" and y.nunique(dropna=True) > 10:
            return "Regression"
        return "Classification"

    def _get_estimator(self, model_name: str, problem_type: str):
        registry = self.MODEL_REGISTRY.get(problem_type, {})
        estimator = registry.get(model_name)
        if estimator is None:
            raise ValueError(f"Unsupported model for {problem_type}: {model_name}")
        return estimator

    def _determine_search_strategy(self, model_name: str, space: Dict[str, List[Any]]) -> str:
        total_choices = 1
        for values in space.values():
            total_choices *= len(values)
            if total_choices > 50:
                return "randomized"
        return "grid"

    MAX_HPO_ITERATIONS = 10

    def _dataset_iterations(self, n_samples: int) -> int:
        if n_samples <= 200:
            return 10
        if n_samples <= 1000:
            return 8
        return 5

    def _build_param_space(self, model_name: str) -> Dict[str, List[Any]]:
        return self.PARAM_GRID.get(model_name, {})

    def _score_default(self, problem_type: str, y_true, y_pred) -> float:
        if problem_type == "Classification":
            return accuracy_score(y_true, y_pred)
        return r2_score(y_true, y_pred)

    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        problem_type: Optional[str] = None,
        cv: int = 3,
        timeout: Optional[int] = None,
        n_iter: Optional[int] = None,
        use_history: bool = True,
    ) -> Dict[str, Any]:
        """Optimize hyperparameters for the requested model."""
        if problem_type is None:
            problem_type = self.detect_problem_type(y)

        try:
            estimator = self._get_estimator(model_name, problem_type)
        except ValueError as error:
            return {
                "status": "failed",
                "diagnostics": str(error),
                "recommendation": "Select a supported model or correct the problem type.",
            }

        param_space = self._build_param_space(model_name)
        if not param_space:
            return {
                "status": "failed",
                "diagnostics": f"No hyperparameter space defined for {model_name}.",
                "recommendation": "Use a supported model with a defined search space.",
            }

        n_samples = len(X)
        if n_iter is None:
            n_iter = self._dataset_iterations(n_samples)
        n_iter = max(1, min(int(n_iter), self.MAX_HPO_ITERATIONS))

        search_method = self._determine_search_strategy(model_name, param_space)
        searcher = None
        try:
            if search_method == "grid":
                searcher = GridSearchCV(
                    estimator=estimator,
                    param_grid=param_space,
                    scoring=self.SCORING[problem_type][0],
                    cv=cv,
                    n_jobs=-1,
                    error_score="raise",
                )
            elif HAS_BAYES:
                searcher = BayesSearchCV(
                    estimator=estimator,
                    search_spaces=param_space,
                    scoring=self.SCORING[problem_type][0],
                    cv=cv,
                    n_iter=n_iter,
                    n_jobs=-1,
                    random_state=42,
                )
            else:
                searcher = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=param_space,
                    scoring=self.SCORING[problem_type][0],
                    cv=cv,
                    n_iter=n_iter,
                    n_jobs=-1,
                    random_state=42,
                    error_score="raise",
                )

            if timeout and hasattr(searcher, "fit") and not hasattr(searcher, "timeout") and hasattr(searcher, "n_iter"):
                # use a practical limit by lowering n_iter for budget
                searcher.n_iter = min(searcher.n_iter, max(1, timeout // 2), self.MAX_HPO_ITERATIONS)

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
            )
            baseline_model = estimator
            baseline_model.fit(X_train, y_train)
            baseline_pred = baseline_model.predict(X_test)
            baseline_score = self._score_default(problem_type, y_test, baseline_pred)

            t0 = datetime.datetime.utcnow()
            searcher.fit(X_train, y_train)
            duration = (datetime.datetime.utcnow() - t0).total_seconds()
            best_estimator = searcher.best_estimator_
            optimized_pred = best_estimator.predict(X_test)
            optimized_score = self._score_default(problem_type, y_test, optimized_pred)

            performance = self.compare_before_after(baseline_score, optimized_score, model_name)
            report = self.generate_report(
                model_name=model_name,
                dataset_size=n_samples,
                parameters_tested=param_space,
                best_params=searcher.best_params_,
                original_score=baseline_score,
                optimized_score=optimized_score,
                training_duration=duration,
                problem_type=problem_type,
                recommendation=performance["recommended_model"],
            )

            if use_history:
                self._learn_from_history(model_name, problem_type)

            self.memory_agent.log_experiment(
                dataset_name=getattr(X, "name", "dataset"),
                dataset_shape=[n_samples, X.shape[1]],
                problem_type=problem_type,
                algorithm_name=model_name,
                hyperparameters=searcher.best_params_,
                train_score=float(baseline_score),
                test_score=float(optimized_score),
                cv_score=float(np.max(searcher.cv_results_["mean_test_score"])) if "mean_test_score" in searcher.cv_results_ else None,
                training_time=duration,
                feature_count=X.shape[1],
                feature_engineering_steps=[],
                data_cleaning_steps=[],
                notes="Hyperparameter optimization run.",
            )

            return {
                "status": "success",
                "model_name": model_name,
                "problem_type": problem_type,
                "baseline_score": baseline_score,
                "optimized_score": optimized_score,
                "improvement": performance["improvement"],
                "best_params": searcher.best_params_,
                "training_duration": duration,
                "report": report,
            }
        except Exception as error:
            diagnostics = traceback.format_exc()
            error_analysis = self.self_healing_agent.analyze_error(str(error), {"training_size": n_samples})
            recovery = self.self_healing_agent.recommend_fix(error_analysis)
            self.self_healing_agent._log_failure(str(error), error_analysis, recovery, success=False)
            return {
                "status": "failed",
                "diagnostics": str(error),
                "stacktrace": diagnostics,
                "error_analysis": error_analysis,
                "recovery": recovery,
            }

    def compare_before_after(self, original_score: float, optimized_score: float, model_name: str) -> Dict[str, str]:
        improvement_value = optimized_score - original_score
        improvement_pct = f"{improvement_value * 100:.1f}%"
        return {
            "original_score": original_score,
            "optimized_score": optimized_score,
            "improvement": f"{improvement_pct if improvement_value >= 0 else '-' + improvement_pct[1:]}",
            "recommended_model": model_name,
        }

    def _learn_from_history(self, model_name: str, problem_type: str) -> None:
        _ = self.memory_agent.learn_from_history()

    def generate_report(
        self,
        model_name: str,
        dataset_size: int,
        parameters_tested: Dict[str, List[Any]],
        best_params: Dict[str, Any],
        original_score: float,
        optimized_score: float,
        training_duration: float,
        problem_type: str,
        recommendation: str,
    ) -> Dict[str, Any]:
        return {
            "model_name": model_name,
            "dataset_size": dataset_size,
            "problem_type": problem_type,
            "parameters_tested": parameters_tested,
            "best_parameters": best_params,
            "original_score": original_score,
            "optimized_score": optimized_score,
            "training_duration": training_duration,
            "recommendation": recommendation,
        }
