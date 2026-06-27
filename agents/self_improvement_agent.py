from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import time
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error

from config import SMART_MODE_MAX_MODEL_JOBS, SMART_MODE_HPO_MAX_ITER, SMART_MODE_HPO_SKIP_ROWS
from agents.hyperparameter_agent import HyperparameterOptimizationAgent
try:
    from agents.ai_ethics_agent import AIEthicsAgent
except Exception:
    AIEthicsAgent = None

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False


class SelfImprovementAgent:
    """Autonomous self-improvement agent: evaluates and improves models."""

    def __init__(self, random_state: int = 42, n_iter_search: int = 20) -> None:
        self.random_state = random_state
        self.n_iter_search = n_iter_search
        self.hyper_agent = HyperparameterOptimizationAgent()

    def _build_preprocessor(self, X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
        numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("var", VarianceThreshold(0.0)),
        ])

        # Use sparse_output where available for compatibility across sklearn versions
        try:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
        except TypeError:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
        )

        return preprocessor, numeric_features, categorical_features

    def evaluate_model(self, model, X: pd.DataFrame, y: pd.Series, problem_type: str = "Classification", cv: int = 5) -> Dict[str, float]:
        scores = {}
        problem_type_normalized = "Classification" if "classification" in problem_type.lower() else "Regression"
        
        if problem_type_normalized == "Classification":
            try:
                y_pred = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
                scores["accuracy"] = float(np.mean(y_pred))
            except Exception:
                scores["accuracy"] = 0.0
            try:
                scores["f1"] = float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="f1_macro")))
            except Exception:
                scores["f1"] = 0.0
            try:
                unique_classes = len(set(y))
                if unique_classes > 2:
                    scores["roc_auc"] = float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="roc_auc_ovr")))
                else:
                    scores["roc_auc"] = float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="roc_auc")))
            except Exception:
                scores["roc_auc"] = 0.0
        else:  # Regression
            try:
                scores["r2"] = float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="r2")))
            except Exception:
                scores["r2"] = 0.0
            try:
                scores["mae"] = float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="neg_mean_absolute_error")))
            except Exception:
                scores["mae"] = 0.0
            try:
                scores["rmse"] = float(np.mean(cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")))
            except Exception:
                scores["rmse"] = 0.0
        
        return scores

    def detect_fit_issues(self, model, X_train, y_train, X_test, y_test) -> Dict[str, Any]:
        issues = {"overfitting": False, "underfitting": False, "train_score": None, "test_score": None}
        try:
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            train_score = f1_score(y_train, train_pred, average="macro")
            test_score = f1_score(y_test, test_pred, average="macro")
            issues["train_score"] = float(train_score)
            issues["test_score"] = float(test_score)
            if train_score - test_score > 0.15:
                issues["overfitting"] = True
            if test_score < 0.4 and train_score < 0.5:
                issues["underfitting"] = True
        except Exception:
            pass
        return issues

    def automatic_feature_engineering(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # Simple correlation removal and univariate selection
        report: Dict[str, Any] = {}
        X_proc = X.copy()
        try:
            # Drop highly correlated features
            corr = X_proc.select_dtypes(include=["number"]).corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            X_proc = X_proc.drop(columns=to_drop, errors="ignore")
            report["dropped_correlated_features"] = to_drop
        except Exception:
            report["dropped_correlated_features"] = []
        return X_proc, report

    def optimize_hyperparameters(
        self,
        estimator,
        param_distributions: Dict[str, Any],
        X,
        y,
        problem_type: str = "Classification",
        cv: int = 3,
        max_seconds: Optional[int] = None,
        n_iter: Optional[int] = None,
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {"best_params": None, "best_score": None}
        problem_type_normalized = "Classification" if "classification" in problem_type.lower() else "Regression"
        scoring_metric = "f1_macro" if problem_type_normalized == "Classification" else "r2"
        if n_iter is None:
            n_iter = self.n_iter_search
        if max_seconds is not None and max_seconds <= 15:
            report["skipped_due_to_budget"] = True
            return report
        if max_seconds is not None and len(X) < SMART_MODE_HPO_SKIP_ROWS:
            report["skipped_due_to_dataset_size"] = True
            report["reason"] = f"Dataset under {SMART_MODE_HPO_SKIP_ROWS} rows; hyperparameter tuning is skipped for Smart Mode."
            return report
        if max_seconds is not None:
            n_iter = max(1, min(n_iter, max_seconds // 3))

        try:
            if HAS_OPTUNA:
                # Use Optuna to optimize hyperparameters (basic wrapper)
                def objective(trial):
                    params = {}
                    for k, v in (param_distributions or {}).items():
                        if isinstance(v, list):
                            params[k] = trial.suggest_categorical(k, v)
                        elif isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                            params[k] = trial.suggest_float(k, v[0], v[1])
                        else:
                            params[k] = v[0] if isinstance(v, list) and v else None
                    try:
                        estimator.set_params(**params)
                    except Exception:
                        pass
                    scores = cross_val_score(estimator, X, y, cv=cv, scoring=scoring_metric)
                    return float(scores.mean())

                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=min(n_iter, 10))
                report["best_params"] = study.best_params
                report["best_score"] = float(study.best_value)
                return report
            else:
                search = RandomizedSearchCV(
                    estimator,
                    param_distributions,
                    n_iter=min(n_iter, 20),
                    cv=cv,
                    random_state=self.random_state,
                    n_jobs=-1,
                    scoring=scoring_metric,
                )
                search.fit(X, y)
                report["best_params"] = search.best_params_
                report["best_score"] = float(search.best_score_)
                return report
        except Exception:
            return report

    def improve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        baseline_model,
        baseline_metrics: Dict[str, float],
        problem_type: str = "Classification",
        extras: Optional[Dict[str, Any]] = None,
        max_iterations: int = 3,
        max_seconds: Optional[int] = None,
        candidate_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run iterative improvement loops and return structured results.

        The method will attempt up to `max_iterations` improvement cycles. It only
        records and returns improved models (relative to baseline_metrics).
        
        Args:
            X: Features dataframe
            y: Target series
            baseline_model: The baseline model to improve upon
            baseline_metrics: Baseline metrics dict
            problem_type: "Classification" or "Regression" (CRITICAL for model selection)
            extras: Optional dict with X_test, y_test for holdout validation
            max_iterations: Number of improvement iterations to attempt
            max_seconds: Time budget for improvement stage in seconds
            candidate_limit: Maximum number of candidate models to consider
        """
        extras = extras or {}
        problem_type_normalized = "Classification" if "classification" in problem_type.lower() else "Regression"
        
        improvement_history: List[Dict[str, Any]] = []
        model_comparison: List[Dict[str, Any]] = []
        validation_history: List[Dict[str, Any]] = []
        optimization_iterations = 0
        improvement_start = time.time()

        if max_seconds is not None and max_seconds <= 15:
            return {
                "model_comparison": [],
                "improvement_history": [],
                "best_model_version": None,
                "best_model": baseline_model,
                "optimization_report": {"skipped_due_to_budget": True},
                "validation_results": [],
                "optimization_iterations": 0,
            }

        preprocessor, num_feats, cat_feats = self._build_preprocessor(X)
        try:
            X_transformed = preprocessor.fit_transform(X)
        except Exception:
            X_transformed = X.fillna(0).values if hasattr(X, 'fillna') else X.values

        # baseline
        baseline_eval = baseline_metrics or {}
        model_comparison.append({"version": "baseline", "metrics": baseline_eval})
        best_overall = {"name": "baseline", "model": baseline_model, "metrics": baseline_eval}

        # Candidate estimators - select based on problem type
        candidates = []
        try:
            model_names = self.hyper_agent.supported_models(problem_type_normalized)
            if candidate_limit is not None:
                model_names = model_names[:candidate_limit]
            for mn in model_names:
                try:
                    est = self.hyper_agent._get_estimator(mn, problem_type_normalized)
                    param_dist = self.hyper_agent.PARAM_GRID.get(mn, {})
                    candidates.append((mn, est, param_dist))
                except Exception:
                    continue
        except Exception:
            candidates = []

        # ethics agent if available
        ethics_agent = AIEthicsAgent() if AIEthicsAgent is not None else None

        for iteration in range(max_iterations):
            improved_this_round = False
            optimization_iterations += 1
            for name, estimator, param_dist in candidates:
                if max_seconds is not None and time.time() - improvement_start >= max_seconds - 5:
                    break
                try:
                    pipe = Pipeline([("est", estimator)])
                    time_left = None
                    if max_seconds is not None:
                        time_left = max(1, int(max_seconds - (time.time() - improvement_start)))
                    opt_report = self.optimize_hyperparameters(
                        pipe,
                        param_dist or {},
                        X_transformed,
                        y,
                        problem_type_normalized,
                        max_seconds=time_left,
                        n_iter=min(self.n_iter_search, SMART_MODE_HPO_MAX_ITER),
                    )
                    best_params = opt_report.get("best_params") or {}
                    try:
                        pipe.set_params(**best_params)
                    except Exception:
                        pass
                    scores = self.evaluate_model(pipe, X_transformed, y, problem_type_normalized)
                    model_comparison.append({"version": f"iter{iteration+1}_{name}", "metrics": scores, "opt_report": opt_report})
                    improvement_history.append({"iteration": iteration + 1, "model": name, "changes": {"hyperparams": opt_report}})

                    # validation on holdout
                    val = {}
                    X_holdout = extras.get("X_test")
                    y_holdout = extras.get("y_test")
                    if X_holdout is not None and y_holdout is not None:
                        try:
                            pipe.fit(X_transformed, y)
                            preds = pipe.predict(X_holdout)
                            
                            if problem_type_normalized == "Classification":
                                val = {
                                    "accuracy": float(accuracy_score(y_holdout, preds)),
                                    "precision": float(precision_score(y_holdout, preds, average="macro", zero_division=0)),
                                    "recall": float(recall_score(y_holdout, preds, average="macro", zero_division=0)),
                                    "f1": float(f1_score(y_holdout, preds, average="macro", zero_division=0)),
                                }
                                try:
                                    val["roc_auc"] = float(roc_auc_score(y_holdout, pipe.predict_proba(X_holdout)[:, 1]))
                                except Exception:
                                    val["roc_auc"] = None
                            else:  # Regression
                                val = {
                                    "r2": float(r2_score(y_holdout, preds)),
                                    "mae": float(mean_absolute_error(y_holdout, preds)),
                                    "rmse": float(np.sqrt(mean_squared_error(y_holdout, preds))),
                                }
                        except Exception:
                            val = {}
                    validation_history.append({"version": f"iter{iteration+1}_{name}", "validation": val})

                    trust = {"fairness_score": None, "trust_score": None}
                    if ethics_agent is not None and val and problem_type_normalized == "Classification":
                        try:
                            preds = pipe.predict(X_holdout) if X_holdout is not None else []
                            fairness_report = ethics_agent.evaluate_model_fairness(list(preds) if hasattr(preds, 'tolist') else list(preds), list(y_holdout) if y_holdout is not None else list(y), {})
                            trust_score = ethics_agent.calculate_ai_governance_score(ethics_agent.analyze_dataset_bias(X), fairness_report, ethics_agent.analyze_privacy_risk(X)).get("score", 0)
                            trust = {"fairness_score": fairness_report.get("fairness_score"), "trust_score": trust_score}
                        except Exception:
                            pass

                    if problem_type_normalized == "Classification":
                        current_metric = scores.get("f1", 0.0)
                        best_metric = best_overall.get("metrics", {}).get("f1", 0.0)
                    else:
                        current_metric = scores.get("r2", 0.0)
                        best_metric = best_overall.get("metrics", {}).get("r2", 0.0)
                    
                    if current_metric > best_metric + 0.005:
                        best_overall = {"name": f"v{iteration+2}_{name}", "model": pipe, "metrics": scores, "opt_report": opt_report, "trust": trust}
                        improved_this_round = True
                except Exception:
                    continue
            if max_seconds is not None and time.time() - improvement_start >= max_seconds - 5:
                break
            if not improved_this_round:
                break

        optimization_report = {"iterations": optimization_iterations}

        return {
            "model_comparison": model_comparison,
            "improvement_history": improvement_history,
            "best_model_version": best_overall.get("name"),
            "best_model": best_overall.get("model"),
            "optimization_report": optimization_report,
            "validation_results": validation_history,
            "optimization_iterations": optimization_iterations,
        }
