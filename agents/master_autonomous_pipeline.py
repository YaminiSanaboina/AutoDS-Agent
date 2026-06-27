from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

from agents.agent_memory_database import AgentMemoryDatabase
from agents.chief_data_scientist_agent import ChiefDataScientistAgent
from agents.dataset_intelligence_agent import DatasetIntelligenceAgent
from agents.deployment_agent import DeploymentAgent
from agents.documentation_agent import DocumentationAgent
from agents.experiment_memory_agent import ExperimentMemoryAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.hyperparameter_agent import HyperparameterOptimizationAgent
from agents.model_registry_agent import ModelRegistryAgent
from agents.ai_ethics_agent import AIEthicsAgent
from agents.cleaning_agent import clean_dataset
from agents.eda_agent import generate_eda, generate_eda_insights
from agents.model_agent import preprocess_data, train_models, train_selected_models, detect_problem_type, get_model_registry
from agents.xai_agent import generate_shap_explanation, get_feature_importance_ranking
from agents.report_agent import generate_pdf_report
from agents.trust_score_calculator import TrustScoreCalculator, create_executive_metrics_object
from utils.health_score import detect_data_issues
from utils.safe_checks import coalesce_dict, coalesce_list, feature_importance_as_dict, normalize_recommendations, safe_dict_get
from agents.self_improvement_agent import SelfImprovementAgent
from config import (
    SMART_MODE_DEFAULT,
    SMART_MODE_BUDGET_SECONDS,
    SMART_MODE_HPO_MAX_ITER,
    SMART_MODE_MAX_MODEL_JOBS,
    SMART_MODE_SHAP_MAX_SAMPLES,
    SMART_MODE_SHAP_DISABLE_ROWS,
    SMART_MODE_HPO_SKIP_ROWS,
    SMART_MODE_CACHE_DIR,
    SMART_MODE_MODEL_CANDIDATES,
    SMART_MODE_REGRESSION_CANDIDATES,
)

try:
    from agents.specialized_agents import XAIAgent
except ImportError:
    XAIAgent = None


class MasterAutonomousPipeline:
    """Main autonomous pipeline orchestrating the AutoDS AI Data Scientist."""

    def __init__(self) -> None:
        self.dataset_agent = DatasetIntelligenceAgent
        self.feature_agent = FeatureEngineeringAgent
        self.chief_agent = ChiefDataScientistAgent()
        self.hyperparameter_agent = HyperparameterOptimizationAgent()
        self.memory_agent = ExperimentMemoryAgent()
        self.agent_memory_db = AgentMemoryDatabase()
        self.model_registry = ModelRegistryAgent()
        self.deployment_agent = DeploymentAgent()
        self.documentation_agent = DocumentationAgent()
        self.ethics_agent = AIEthicsAgent()
        self.xai_agent = XAIAgent()

    def _normalize_problem_type(self, problem_type: str) -> str:
        if not isinstance(problem_type, str):
            return "Classification"

        normalized = problem_type.strip().lower()
        if "classification" in normalized or "class" in normalized:
            return "Classification"
        if "regression" in normalized or "regress" in normalized:
            return "Regression"
        return "Classification"

    def _validate_model_problem_type(self, model_name: str, problem_type: str) -> bool:
        """Validate that a model type matches the problem type."""
        problem_type_normalized = self._normalize_problem_type(problem_type)
        
        classification_models = {
            "Logistic Regression", "Random Forest", "Decision Tree", "SVM",
            "Gradient Boosting", "XGBoost", "LightGBM", "CatBoost",
            "Extra Trees", "AdaBoost", "KNN", "Naive Bayes",
            "RandomForestClassifier", "DecisionTreeClassifier", "SVC", "GradientBoostingClassifier",
            "XGBClassifier", "LightGBMClassifier", "CatBoostClassifier",
        }
        
        regression_models = {
            "Linear Regression", "Random Forest Regressor", "Decision Tree Regressor", "SVR",
            "Gradient Boosting Regressor", "XGBoost Regressor", "LightGBM Regressor", "CatBoost Regressor",
            "Ridge", "Lasso", "ElasticNet", "Extra Trees Regressor",
            "RandomForestRegressor", "DecisionTreeRegressor", "GradientBoostingRegressor",
            "XGBRegressor", "LightGBMRegressor", "CatBoostRegressor",
        }
        
        if problem_type_normalized == "Classification":
            if model_name in regression_models:
                _logger.warning(f"[VALIDATION WARNING] Regression model '{model_name}' selected for Classification problem!")
                return False
        else:  # Regression
            if model_name in classification_models:
                _logger.warning(f"[VALIDATION WARNING] Classification model '{model_name}' selected for Regression problem!")
                return False
        
        return True

    def _build_hyperparameter_overview(self, algorithm: str, problem_type: str) -> Dict[str, Any]:
        problem_type_key = self._normalize_problem_type(problem_type)
        supported_models = self.hyperparameter_agent.supported_models(problem_type_key)
        recommended_parameters = self.hyperparameter_agent.PARAM_GRID.get(algorithm, {})
        return {
            "algorithm": algorithm,
            "problem_type": problem_type_key,
            "supported_models": supported_models,
            "recommended_parameters": recommended_parameters,
            "notes": (
                "Hyperparameter tuning plan based on model selection and dataset problem type. "
                "Use the supported model grid to guide optimization."
            ),
        }

    def _score_deployment_readiness(self, deployment_readiness: Dict[str, Any]) -> int:
        risk_level = deployment_readiness.get("risk_level", "Medium")
        score_map = {"Low": 90, "Medium": 60, "High": 30}
        return score_map.get(risk_level, 50)

    def _record_stage_error(self, stage: str, error: Exception, stage_errors: List[Dict[str, Any]]) -> None:
        error_text = str(error)
        if not error_text:
            error_text = f"{type(error).__name__}: An error occurred during {stage}."

        entry = {
            "stage": stage,
            "error": error_text,
            "traceback": traceback.format_exc(),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }
        try:
            from agents.self_healing_agent import SelfHealingAgent

            healer = SelfHealingAgent()
            analysis = healer.analyze_error(str(error))
            fix = healer.recommend_fix(analysis)
            entry["recovery"] = fix
            entry["user_message"] = fix.get("explanation") or analysis.get("root_cause")
        except Exception:
            pass
        stage_errors.append(entry)

    @staticmethod
    def _has_explainability_feature_importance(feature_importance: Any) -> bool:
        """Safe non-empty check for dict, DataFrame, Series, or other feature importance payloads."""
        from utils.safe_checks import is_present

        return is_present(feature_importance)

    def _derive_feature_importance(
        self,
        model: Any,
        extras: Dict[str, Any],
        feature_names: List[str],
        X: Optional[pd.DataFrame] = None,
        y: Optional[Any] = None,
    ) -> Dict[str, float]:
        """Fallback feature importance when SHAP is unavailable."""
        importance: Dict[str, float] = {}

        try:
            if isinstance(extras, dict):
                importance = feature_importance_as_dict(extras.get("feature_importance"))
        except Exception:
            importance = {}

        if not importance:
            try:
                if hasattr(model, "feature_importances_") and model.feature_importances_ is not None:
                    importance = dict(zip(feature_names, model.feature_importances_.tolist()))
            except Exception:
                importance = {}

        if not importance:
            try:
                if hasattr(model, "coef_") and model.coef_ is not None:
                    coef = model.coef_
                    if getattr(coef, "ndim", 1) > 1:
                        coef = np.abs(coef).mean(axis=0)
                    else:
                        coef = np.abs(coef)
                    importance = dict(zip(feature_names, np.asarray(coef).flatten().tolist()))
            except Exception:
                importance = {}

        if not importance and X is not None and y is not None:
            try:
                from sklearn.inspection import permutation_importance

                X_sample = X
                y_sample = y
                if hasattr(X, "shape") and X.shape[0] > 100:
                    sample_indices = X.sample(n=100, random_state=42).index
                    X_sample = X.loc[sample_indices]
                    if hasattr(y, "loc"):
                        y_sample = y.loc[sample_indices]
                    else:
                        y_sample = [y[i] for i in sample_indices]

                perm_result = permutation_importance(
                    model,
                    X_sample,
                    y_sample,
                    n_repeats=3,
                    random_state=42,
                    n_jobs=1,
                )
                importance = dict(zip(feature_names, perm_result.importances_mean.tolist()))
            except Exception:
                importance = {}

        cleaned: Dict[str, float] = {}
        for key, value in (importance or {}).items():
            if key is None:
                continue
            try:
                cleaned[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return cleaned

    def _replay_cache_progress(self, progress_callback: Optional[Callable[[Dict[str, Any]], None]]) -> None:
        """Emit completed stage events when serving a cached pipeline result."""
        if not progress_callback:
            return
        for stage, pct in (
            ("dataset_upload", 2),
            ("dataset_intelligence", 10),
            ("data_cleaning", 20),
            ("eda", 30),
            ("feature_engineering", 35),
            ("automl", 60),
            ("model_comparison", 70),
            ("explainability", 80),
            ("ai_ethics_trust", 88),
            ("deployment_readiness", 92),
            ("pdf_report", 100),
        ):
            progress_callback({"stage": stage, "status": "completed", "percent": pct})

    def _notify_progress(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        stage: str,
        status: str,
        percent: int,
        result: Any = None,
    ) -> None:
        if not progress_callback:
            return
        payload: Dict[str, Any] = {"stage": stage, "status": status, "percent": percent}
        if result is not None:
            payload["result"] = result
        progress_callback(payload)

    def _is_value_present(self, value: Any) -> bool:
        from utils.safe_checks import is_present

        return is_present(value)

    def run_pipeline(
        self,
        dataset: pd.DataFrame,
        dataset_name: str,
        project_goal: str,
        constraints: Optional[Dict[str, Any]] = None,
        smart_mode: Optional[bool] = None,
        max_seconds: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Run the master autonomous AI data scientist pipeline."""
        constraints = constraints or {}
        # Smart Mode defaults and timing
        smart_mode = SMART_MODE_DEFAULT if smart_mode is None else bool(smart_mode)
        max_seconds = int(SMART_MODE_BUDGET_SECONDS if max_seconds is None else int(max_seconds))
        start_time = datetime.datetime.utcnow().isoformat() + "Z"

        # Prepare cache for smart mode: basic fingerprint
        try:
            os.makedirs(SMART_MODE_CACHE_DIR, exist_ok=True)
        except Exception:
            pass
        try:
            sample_csv = dataset.head(100).to_csv(index=False).encode("utf-8")
            ds_hash = hashlib.md5(sample_csv + str(dataset.shape).encode()).hexdigest()
        except Exception:
            ds_hash = hashlib.md5(str(dataset.shape).encode()).hexdigest()
        cache_path = os.path.join(SMART_MODE_CACHE_DIR, f"autopipeline_{dataset_name}_{ds_hash}.json")
        if smart_mode and use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as fh:
                    cached_output = json.load(fh)
                self._replay_cache_progress(progress_callback)
                return cached_output
            except Exception:
                pass

        run_start = time.time()
        stage_durations: Dict[str, float] = {}
        stage_errors: List[Dict[str, Any]] = []
        model_comparison: List[Any] = []
        improvement_history: List[Any] = []
        best_model_version = None
        model_registry_entry = None
        optimization_report: Dict[str, Any] = {}
        validation_results: Dict[str, Any] = {}
        model_versions: List[Any] = []
        production_model = None
        deployment_readiness: Dict[str, Any] = {"risk_level": "Unknown"}
        target = None
        best_model = None
        best_name = None
        results: Dict[str, Any] = {}
        X = None
        y = None
        extras: Dict[str, Any] = {}
        target_encoder = None
        cleaned_df = dataset.copy()
        training_artifacts: Dict[str, Any] = {}

        self._notify_progress(progress_callback, "dataset_upload", "completed", 2, {"dataset_name": dataset_name})

        dataset_agent = DatasetIntelligenceAgent(dataset, dataset_name)
        # run dataset intelligence
        if progress_callback:
            progress_callback({"stage": "dataset_intelligence", "status": "started", "percent": 5})
        dataset_start = time.time()
        dataset_report = dataset_agent.generate_dataset_report()
        dataset_end = time.time()
        stage_durations["dataset_analysis"] = dataset_end - dataset_start
        if progress_callback:
            progress_callback({"stage": "dataset_intelligence", "status": "completed", "percent": 10, "result": dataset_report})
        _logger.debug(f"[MasterPipeline] dataset_report keys: {list(dataset_report.keys()) if isinstance(dataset_report, dict) else type(dataset_report)}")

        feature_agent = FeatureEngineeringAgent(dataset, dataset_name)
        feature_start = time.time()
        feature_plan = feature_agent.generate_feature_plan()
        feature_end = time.time()
        stage_durations["feature_engineering"] = feature_end - feature_start
        self._notify_progress(progress_callback, "feature_engineering", "started", 22)
        if progress_callback:
            progress_callback({"stage": "feature_engineering", "status": "completed", "percent": 25, "result": feature_plan})
        _logger.debug(f"[MasterPipeline] feature_plan keys: {list(feature_plan.keys()) if isinstance(feature_plan, dict) else type(feature_plan)}")
        suggestion_tags = ["feature_engineering", dataset_report["problem_analysis"]["problem_type"]]

        project_summary = self.chief_agent.analyze_project(
            dataset=dataset,
            dataset_metadata={
                "name": dataset_name,
                "goal": project_goal,
                "constraints": constraints,
                "missing_percent": float(dataset.isnull().mean().mean() * 100),
                "duplicate_percent": float(dataset.duplicated().mean() * 100),
            },
            model_history=self.memory_agent.get_history(dataset_name=dataset_name),
        )
        _logger.debug(f"[MasterPipeline] project_summary keys: {list(project_summary.keys()) if isinstance(project_summary, dict) else type(project_summary)}")

        model_strategy = project_summary.get("problem_insights", {})
        recommended_models = dataset_report.get("model_strategy", {}).get("recommendations", [])
        chosen_model = recommended_models[0] if recommended_models else {"model": "Random Forest"}
        algorithm = chosen_model.get("model", "Random Forest")
        problem_type = model_strategy.get("problem_type", dataset_report.get("problem_analysis", {}).get("problem_type", "Classification"))

        hyperparameter_report = self._build_hyperparameter_overview(
            algorithm=algorithm,
            problem_type=problem_type,
        )
        self._notify_progress(
            progress_callback,
            "hyperparameter_optimization",
            "completed",
            18,
            hyperparameter_report,
        )

        # Cleaning stage
        self._notify_progress(progress_callback, "data_cleaning", "started", 15)
        cleaning_start = time.time()
        try:
            cleaned_df, cleaning_report = clean_dataset(dataset)
        except Exception as exc:
            self._record_stage_error("data_cleaning", exc, stage_errors)
            cleaned_df, cleaning_report = dataset.copy(), {"status": "cleaning_failed", "error": str(exc)}
        cleaning_end = time.time()
        cleaning_results = {"report": cleaning_report, "shape": (int(cleaned_df.shape[0]), int(cleaned_df.shape[1]))}
        stage_durations["data_cleaning"] = cleaning_end - cleaning_start
        _logger.debug(f"[MasterPipeline] cleaning_results keys: {list(cleaning_results.keys())}")
        if progress_callback:
            progress_callback({"stage": "data_cleaning", "status": "completed", "percent": 20, "result": cleaning_results})

        # EDA stage (run in parallel where possible)
        eda_raw = {}
        eda_insights = []
        eda_start = time.time()
        try:
            if progress_callback:
                progress_callback({"stage": "eda", "status": "started", "percent": 22})
            from concurrent.futures import ThreadPoolExecutor

            def _run_eda(df):
                try:
                    return generate_eda(df)
                except Exception:
                    return {}

            def _run_eda_insights(df, numerics):
                try:
                    return generate_eda_insights(df, numerics)
                except Exception:
                    return []

            with ThreadPoolExecutor(max_workers=2) as ex:
                f1 = ex.submit(_run_eda, cleaned_df)
                # wait for f1 to get numerical columns
                eda_raw = f1.result()
                numerics = eda_raw.get("numerical_columns", [])
                f2 = ex.submit(_run_eda_insights, cleaned_df, numerics)
                eda_insights = f2.result()
        except Exception as exc:
            self._record_stage_error("eda", exc, stage_errors)
            eda_raw, eda_insights = {}, []
        eda_end = time.time()
        eda_results = {"summary": eda_raw.get("summary"), "numerical_columns": eda_raw.get("numerical_columns", []), "categorical_columns": eda_raw.get("categorical_columns", []), "insights": eda_insights, "charts": {}}
        stage_durations["eda"] = eda_end - eda_start
        _logger.debug(f"[MasterPipeline] eda_results keys: {list(eda_results.keys())}")
        if progress_callback:
            progress_callback({"stage": "eda", "status": "completed", "percent": 30, "result": eda_results})

        # XAI placeholder (populated after model training)
        xai_results = {"explainability": "Not available"}

        ethics_start = time.time()
        try:
            ethics_bias = self.ethics_agent.analyze_dataset_bias(dataset)
            ethics_privacy = self.ethics_agent.analyze_privacy_risk(dataset)
            ethics_report = self.ethics_agent.generate_ethics_report(
                bias_report=ethics_bias,
                fairness_report={
                    "fairness_score": 50.0,
                    "risk_level": "Unknown",
                    "group_metrics": {},
                },
                privacy_report=ethics_privacy,
            )
        except Exception as exc:
            self._record_stage_error("ethics", exc, stage_errors)
            ethics_bias = {}
            ethics_privacy = {}
            ethics_report = {
                "bias_report": {},
                "fairness_report": {},
                "privacy_report": {},
                "status": "ethics_analysis_failed",
                "error": str(exc),
            }
        ethics_end = time.time()
        stage_durations["ethics"] = ethics_end - ethics_start
        if progress_callback:
            progress_callback({"stage": "ai_ethics_trust", "status": "completed", "percent": 35, "result": ethics_report})
        _logger.debug(f"[MasterPipeline] ethics_report keys: {list(ethics_report.keys())}")

        experiment_entry = self.memory_agent.log_experiment(
            dataset_name=dataset_name,
            dataset_shape=[int(dataset.shape[0]), int(dataset.shape[1])],
            problem_type=problem_type,
            algorithm_name=algorithm,
            hyperparameters=hyperparameter_report.get("recommended_parameters", {}),
            train_score=None,
            test_score=None,
            cv_score=None,
            training_time=0.0,
            feature_count=len(dataset.columns),
            feature_engineering_steps=feature_plan.get("steps", []),
            data_cleaning_steps=[step for step in feature_plan.get("steps", []) if "impute" in step.lower() or "remove" in step.lower()],
            notes=f"Project goal: {project_goal}",
        )

        self.agent_memory_db.learn_from_experiment({
            "agent": "HyperparameterOptimizationAgent",
            "title": f"{algorithm} optimization for {dataset_name}",
            "result": {
                "best_metric": None,
                "best_params": hyperparameter_report.get("recommended_parameters", {}),
            },
            "tags": suggestion_tags,
            "importance": 0.8,
            "success_score": 0.7,
        })

        # Model training stage
        # Determine target column
        problem_analysis = dataset_report.get("problem_analysis", {})
        target = problem_analysis.get("likely_target") or (dataset.columns[-1] if len(dataset.columns) > 0 else None)
        model_results = {"best_model": None, "metrics": {}, "model_readiness_score": 0}
        explainability_results = {"shap_values": {}, "feature_importance": {}}
        ai_trust_results = {"fairness_score": 50.0, "bias_analysis": "", "trust_score": 0.0}

        if target is not None:
            try:
                self._notify_progress(progress_callback, "automl", "started", 40)
                X, y, target_encoder = preprocess_data(cleaned_df, target)
                pt = detect_problem_type(y)
                _logger.debug(f"[MasterPipeline] Detected problem type: {pt}")
                model_training_start = time.time()
                # Smart Mode: constrain hyperparameter tuning and candidate models
                try:
                    if smart_mode:
                        try:
                            if hasattr(self.hyperparameter_agent, 'n_iter_search'):
                                self.hyperparameter_agent.n_iter_search = min(getattr(self.hyperparameter_agent, 'n_iter_search', SMART_MODE_HPO_MAX_ITER), SMART_MODE_HPO_MAX_ITER)
                        except Exception:
                            pass

                        elapsed = time.time() - run_start
                        remaining = max_seconds - elapsed if max_seconds is not None else None
                        dataset_rows = int(cleaned_df.shape[0]) if hasattr(cleaned_df, 'shape') else None

                        selected = None
                        try:
                            if pt == "Classification" and SMART_MODE_MODEL_CANDIDATES:
                                selected = list(SMART_MODE_MODEL_CANDIDATES)
                            elif pt != "Classification" and SMART_MODE_REGRESSION_CANDIDATES:
                                selected = list(SMART_MODE_REGRESSION_CANDIDATES)
                            else:
                                registry = get_model_registry(pt, smart_mode=True)
                                selected = list(registry.keys())
                        except Exception:
                            selected = None

                        training_budget = None
                        if remaining is not None:
                            training_budget = max(45, int(remaining - 15))

                        results, best_name, best_model, extras = train_selected_models(
                            X,
                            y,
                            pt,
                            selected_models=selected,
                            max_seconds=training_budget,
                            smart_mode=True,
                        )
                        if dataset_rows is not None and dataset_rows < SMART_MODE_HPO_SKIP_ROWS:
                            optimization_report = {"skipped_due_to_dataset_size": True, "reason": f"Dataset under {SMART_MODE_HPO_SKIP_ROWS} rows, HPO skipped for Smart Mode."}
                        else:
                            optimization_report = {}
                    else:
                        results, best_name, best_model, extras = train_selected_models(X, y, pt, smart_mode=False)
                except Exception as exc:
                    self._record_stage_error("model_training_selection", exc, stage_errors)
                    results, best_name, best_model, extras = train_selected_models(X, y, pt)

                # VALIDATION: Ensure best model matches problem type
                if not self._validate_model_problem_type(best_name, pt):
                    _logger.warning(f"[MasterPipeline] CRITICAL: Model selection mismatch! Best model '{best_name}' does not match problem type '{pt}'")
                    # Log the issue but continue (model was already trained)

                # Persist initial best model (baseline v1)
                try:
                    improvement_agent = SelfImprovementAgent()
                except Exception:
                    improvement_agent = None
                try:
                    # build artifact paths
                    artifact_base = os.path.join("deployment_package", f"DEP_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
                    v1_dir = os.path.join(artifact_base, "v1")
                    os.makedirs(v1_dir, exist_ok=True)
                    model_path = os.path.join(v1_dir, "best_model.pkl")
                    encoder_path = os.path.join(v1_dir, "target_encoder.pkl")
                    preproc_path = os.path.join(v1_dir, "preprocessing_pipeline.pkl")
                    feature_schema_path = os.path.join(v1_dir, "feature_schema.json")
                    metadata_path = os.path.join(v1_dir, "model_metadata.json")

                    joblib.dump(best_model, model_path)
                    if target_encoder is not None:
                        joblib.dump(target_encoder, encoder_path)
                    # try to create a preprocessing pipeline
                    try:
                        if improvement_agent is not None:
                            preproc, _, _ = improvement_agent._build_preprocessor(X)
                            joblib.dump(preproc, preproc_path)
                        else:
                            joblib.dump(None, preproc_path)
                    except Exception:
                        joblib.dump(None, preproc_path)

                    # feature schema
                    try:
                        import json
                        schema = {"features": [str(c) for c in list(X.columns)], "dtypes": {str(c): str(X[c].dtype) for c in X.columns}}
                        with open(feature_schema_path, "w", encoding="utf-8") as f:
                            json.dump(schema, f)
                    except Exception:
                        pass

                    # metadata
                    try:
                        metadata = {
                            "version": "v1",
                            "model_name": best_name,
                            "dataset": dataset_name,
                            "trained_at": datetime.datetime.utcnow().isoformat() + "Z",
                            "metrics": results.get(best_name) if isinstance(results, dict) else None,
                        }
                        import json
                        with open(metadata_path, "w", encoding="utf-8") as f:
                            json.dump(metadata, f)
                    except Exception:
                        pass

                    # update model_registry_entry artifact path if exists
                    try:
                        model_registry_entry["artifact_path"] = model_path
                        model_registry_entry.setdefault("extra_metadata", {})["encoder_path"] = encoder_path
                        model_registry_entry.setdefault("extra_metadata", {})["preprocessing_path"] = preproc_path
                        model_registry_entry.setdefault("extra_metadata", {})["feature_schema"] = feature_schema_path
                    except Exception:
                        pass

                    # initialize model_versions and production pointer
                    model_versions = [{
                        "version": "v1",
                        "artifact_path": model_path,
                        "encoder_path": encoder_path,
                        "preprocessing_path": preproc_path,
                        "feature_schema": feature_schema_path,
                        "metadata_path": metadata_path,
                        "metrics": results.get(best_name) if isinstance(results, dict) else None,
                    }]
                    production_model = {"version": "v1", "artifact_path": model_path}
                except Exception as e:
                    _logger.warning(f"[MasterPipeline] Failed to persist baseline artifacts: {e}")
                    model_versions = []
                    production_model = None
                model_training_end = time.time()
                stage_durations["auto_ml"] = model_training_end - model_training_start
                best_score = float(results.get(best_name, 0.0)) if isinstance(results, dict) else 0.0
                model_results = {
                    "best_model": best_name,
                    "best_score": best_score,
                    "metrics": results,
                    "model_readiness_score": int(best_score * 100) if best_score <= 1 else int(round(best_score)),
                    "problem_type": pt,
                    "training_times": coalesce_dict(extras.get("training_times")),
                    "detailed_metrics": coalesce_dict(extras.get("detailed_metrics")),
                    "model_selection_explanation": coalesce_dict(extras.get("model_selection_explanation")),
                }
                if isinstance(dataset_report, dict):
                    problem_block = coalesce_dict(dataset_report.get("problem_analysis"))
                    problem_block["problem_type"] = pt
                    problem_block["detected_problem_type"] = pt
                    dataset_report["problem_analysis"] = problem_block
                problem_type = pt
                if progress_callback:
                    progress_callback({"stage": "automl", "status": "completed", "percent": 55, "result": model_results})
                training_artifacts = {
                    "best_model": best_model,
                    "best_name": best_name,
                    "results": results,
                    "problem_type": pt,
                    "target_column": target,
                    "X_data": X,
                    "y_data": y,
                    "target_encoder": target_encoder,
                    "extras": extras,
                    "cleaned_dataframe": cleaned_df,
                }
                # Explainability and AI trust/fairness — run in parallel where possible
                try:
                    explainability_start = time.time()
                    remaining = None
                    if smart_mode and max_seconds:
                        elapsed = time.time() - run_start
                        remaining = max_seconds - elapsed

                    explainability_results = {"shap_values": {}, "feature_importance": {}}
                    ai_trust_results = {"fairness_score": 50.0, "bias_analysis": ethics_bias, "trust_score": 0.0}

                    # Initialize fairness computation handle to None; it may be
                    # defined later when SHAP is enabled. Guard calls against None
                    # to avoid UnboundLocalError after recent refactors.
                    _compute_fairness = None

                    def _compute_shap(model, X_in):
                        inp = X_in
                        try:
                            if smart_mode and isinstance(inp, pd.DataFrame) and len(inp) > SMART_MODE_SHAP_MAX_SAMPLES:
                                inp = inp.sample(SMART_MODE_SHAP_MAX_SAMPLES, random_state=42)
                        except Exception:
                            pass
                        return generate_shap_explanation(model, inp)

                    shap_disabled = False
                    try:
                        if smart_mode and isinstance(X, pd.DataFrame) and len(X) > SMART_MODE_SHAP_DISABLE_ROWS:
                            shap_disabled = True
                    except Exception:
                        shap_disabled = False

                    if shap_disabled:
                        # When SHAP is disabled (e.g., Smart Mode with large datasets),
                        # ensure parallel flags and timeouts are defined for later logic.
                        fallback_fi = self._derive_feature_importance(best_model, extras, list(X.columns), X, y)
                        explainability_results = {
                            "shap_values": {},
                            "feature_importance": fallback_fi,
                            "summary": "Feature importance generated using model-native importance (Smart Mode large-dataset path).",
                        }
                        use_parallel = False
                        explainability_timeout = None
                    else:
                        def _compute_fairness(model, X_in, y_in):
                            try:
                                preds = model.predict(X_in) if model is not None else []
                                fairness = self.ethics_agent.evaluate_model_fairness(preds.tolist() if hasattr(preds, 'tolist') else list(preds), list(y_in), {})
                                trust_score = self.ethics_agent.calculate_ai_governance_score(ethics_bias, fairness, ethics_privacy).get("score", 0)
                                return fairness, trust_score
                            except Exception:
                                return {"fairness_score": 50.0, "risk_level": "Unknown"}, 0.0

                        use_parallel = True
                        if remaining is not None and remaining < 10:
                            use_parallel = False

                        from concurrent.futures import ThreadPoolExecutor

                        explainability_timeout = None
                        if remaining is not None:
                            explainability_timeout = max(2, min(int(remaining - 1), 10))

                        if use_parallel:
                            with ThreadPoolExecutor(max_workers=2) as ex:
                                fut_shap = ex.submit(_compute_shap, best_model, extras.get("X_test", X))
                                fut_fair = None
                                if _compute_fairness is not None:
                                    fut_fair = ex.submit(_compute_fairness, best_model, extras.get("X_test", X), extras.get("y_test", y))
                                try:
                                    shap_res = fut_shap.result(timeout=explainability_timeout)
                                    explainer, shap_values = shap_res
                                    fi = get_feature_importance_ranking(shap_values, list(X.columns))
                                    explainability_results = {"shap_values": getattr(shap_values, "values", shap_values), "feature_importance": fi}
                                except Exception as exc:
                                    if isinstance(exc, TimeoutError):
                                        exc = Exception(
                                            f"SHAP explainability timed out after {explainability_timeout} seconds; using fallback feature importance."
                                        )
                                    try:
                                        fut_shap.cancel()
                                    except Exception:
                                        pass
                                    self._record_stage_error("explainability", exc, stage_errors)
                                    _logger.warning(f"[MasterPipeline] SHAP explainability failed: {exc}")
                                if fut_fair is not None:
                                    try:
                                        fairness_report, trust_score = fut_fair.result(timeout=explainability_timeout)
                                        fs = fairness_report.get("fairness_score", 50.0)
                                        if isinstance(fs, str):
                                            fs = TrustScoreCalculator.ensure_numeric_trust_score(fs) or 50.0
                                        ts = trust_score if isinstance(trust_score, (int, float)) else trust_score.get("score", 0) if isinstance(trust_score, dict) else 0
                                        ts = float(ts) if isinstance(ts, (int, float)) else 0
                                        ai_trust_results = {"fairness_score": fs, "bias_analysis": ethics_bias, "trust_score": ts}
                                    except Exception as exc:
                                        self._record_stage_error("ai_trust", exc, stage_errors)
                                        _logger.warning(f"[MasterPipeline] Trust analysis failed or timed out: {exc}")
                                        ai_trust_results = {"fairness_score": 50.0, "bias_analysis": ethics_bias, "trust_score": 0.0}
                                else:
                                    ai_trust_results = {"fairness_score": 50.0, "bias_analysis": ethics_bias, "trust_score": 0.0}
                        else:
                            try:
                                explainer, shap_values = _compute_shap(best_model, extras.get("X_test", X))
                                fi = get_feature_importance_ranking(shap_values, list(X.columns))
                                explainability_results = {"shap_values": getattr(shap_values, "values", shap_values), "feature_importance": fi}
                            except Exception as exc:
                                self._record_stage_error("explainability", exc, stage_errors)
                                _logger.warning(f"[MasterPipeline] SHAP explainability failed: {exc}")
                            try:
                                fairness_report, trust_score = _compute_fairness(best_model, extras.get("X_test", X), extras.get("y_test", y))
                                fs = fairness_report.get("fairness_score", 50.0)
                                if isinstance(fs, str):
                                    fs = TrustScoreCalculator.ensure_numeric_trust_score(fs) or 50.0
                                ts = trust_score if isinstance(trust_score, (int, float)) else trust_score.get("score", 0) if isinstance(trust_score, dict) else 0
                                ts = float(ts) if isinstance(ts, (int, float)) else 0
                                ai_trust_results = {"fairness_score": fs, "bias_analysis": ethics_bias, "trust_score": ts}
                            except Exception as exc:
                                self._record_stage_error("ai_trust", exc, stage_errors)
                                _logger.warning(f"[MasterPipeline] Trust analysis failed: {exc}")

                    feature_importance_data = explainability_results.get("feature_importance")
                    feature_importance_empty = (
                        feature_importance_data is None
                        or (hasattr(feature_importance_data, "__len__") and len(feature_importance_data) == 0)
                    )
                    if feature_importance_empty:
                        fallback_fi = self._derive_feature_importance(best_model, extras, list(X.columns), X, y)
                        if fallback_fi:
                            explainability_results = {
                                "shap_values": {},
                                "feature_importance": fallback_fi,
                                "summary": "SHAP explainability failed; using fallback feature importance.",
                            }
                        else:
                            explainability_results = {
                                "shap_values": {},
                                "feature_importance": {},
                                "summary": "Explainability is unavailable for this model.",
                            }
                except Exception as e:
                    self._record_stage_error("explainability", e, stage_errors)
                    _logger.warning(f"[MasterPipeline] Explainability/Fairness failed: {e}")
                explainability_end = time.time()
                stage_durations["explainability"] = explainability_end - explainability_start
                if progress_callback:
                    progress_callback({"stage": "explainability", "status": "completed", "percent": 70, "result": explainability_results})
                # Deployment readiness (mlops)
                try:
                    test_score_val = results.get(best_name) if isinstance(results, dict) else None
                    # Ensure hyperparameter_report is a dict, not a DataFrame
                    hpr = hyperparameter_report.get("recommended_parameters", {}) if isinstance(hyperparameter_report, dict) else {}
                    if hasattr(hpr, "to_dict"):
                        hpr = hpr.to_dict()
                    elif not isinstance(hpr, dict):
                        hpr = {}
                    
                    # Check explainability_available safely (it might be a DataFrame or dict)
                    exp_available = False
                    if isinstance(explainability_results, dict):
                        feat_imp = explainability_results.get("feature_importance")
                        if feat_imp is not None:
                            if isinstance(feat_imp, dict):
                                exp_available = bool(len(feat_imp) > 0)
                            elif hasattr(feat_imp, "__len__"):
                                try:
                                    exp_available = len(feat_imp) > 0
                                except (ValueError, TypeError):
                                    exp_available = False
                            elif hasattr(feat_imp, "empty"):
                                try:
                                    exp_available = not feat_imp.empty
                                except (ValueError, TypeError):
                                    exp_available = False
                            else:
                                from utils.safe_checks import safe_bool
                                exp_available = safe_bool(feat_imp)
                    
                    deployment_readiness = self.deployment_agent.analyze_deployment_risk({
                        "metrics": {"test_score": test_score_val}, 
                        "training_info": {
                            "preprocessing_steps": feature_plan.get("steps", []) if isinstance(feature_plan, dict) else [], 
                            "explainability_available": exp_available
                        }, 
                        "feature_list": list(X.columns) if hasattr(X, "columns") else [], 
                        "hyperparameters": hpr
                    })
                except Exception as exc:
                    self._record_stage_error("deployment_readiness", exc, stage_errors)
                    deployment_readiness = {"risk_level": "Unknown", "warnings": [str(exc)]}
                
                if progress_callback:
                    progress_callback({"stage": "deployment_readiness", "status": "completed", "percent": 75, "result": deployment_readiness})

                # Self-improvement step: evaluate and attempt automated improvements
                improvement_start = time.time()
                try:
                    improvement_results = {}
                    if smart_mode:
                        # check remaining time before attempting improvements
                        remaining_time = None
                        try:
                            elapsed = time.time() - run_start
                            remaining_time = max_seconds - elapsed if max_seconds else None
                        except Exception:
                            remaining_time = None
                        if remaining_time is None or remaining_time > 120:
                            improvement_agent = SelfImprovementAgent()
                            improvement_results = improvement_agent.improve(
                                X,
                                y,
                                best_model,
                                results,
                                problem_type=self._normalize_problem_type(problem_type),
                                extras=extras,
                                max_seconds=remaining_time,
                                candidate_limit=1,
                                max_iterations=1,
                            )
                        else:
                            # skip heavy improvements under tight budget
                            improvement_results = {
                                "improvement_history": [],
                                "model_comparison": [],
                                "optimization_report": {"skipped_due_to_budget": True},
                                "best_model": best_model,
                                "best_model_version": best_name,
                            }
                    else:
                        improvement_agent = SelfImprovementAgent()
                        improvement_results = improvement_agent.improve(X, y, best_model, results, problem_type=self._normalize_problem_type(problem_type), extras=extras)
                    # merge improvement outputs
                    model_comparison = improvement_results.get("model_comparison", [])
                    improvement_history = improvement_results.get("improvement_history", [])
                    best_model_version = improvement_results.get("best_model_version")
                    optimization_report = improvement_results.get("optimization_report", {})
                    validation_results = improvement_results.get("validation_results", {})
                    # Persist best model object and encoder if available
                    try:
                        best_model_obj = improvement_results.get("best_model")
                        artifact_dir = os.path.join("deployment_package", f"DEP_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
                        os.makedirs(artifact_dir, exist_ok=True)
                        artifact_model_path = None
                        artifact_encoder_path = None
                        if best_model_obj is not None:
                            artifact_model_path = os.path.join(artifact_dir, "best_model.pkl")
                            joblib.dump(best_model_obj, artifact_model_path)
                        if 'target_encoder' in locals() and target_encoder is not None:
                            artifact_encoder_path = os.path.join(artifact_dir, "target_encoder.pkl")
                            joblib.dump(target_encoder, artifact_encoder_path)
                        # update model_registry_entry artifact_path and extras
                        try:
                            model_registry_entry["artifact_path"] = artifact_model_path
                            model_registry_entry.setdefault("extra_metadata", {})["encoder_path"] = artifact_encoder_path
                        except Exception:
                            pass
                        # expose artifact paths in improvement report
                        optimization_report.setdefault("artifacts", {})
                        optimization_report["artifacts"]["model"] = artifact_model_path
                        optimization_report["artifacts"]["encoder"] = artifact_encoder_path
                    except Exception as e:
                        _logger.warning(f"[MasterPipeline] Failed to persist artifacts: {e}")
                    # if a new best model found, update registry metadata
                    if best_model_version and best_model_version != best_name:
                        model_results["improved_best_model"] = best_model_version
                        if model_registry_entry:
                            try:
                                model_registry_entry["version"] = best_model_version
                            except Exception:
                                pass
                except Exception as e:
                    self._record_stage_error("self_improvement", e, stage_errors)
                    _logger.warning(f"[MasterPipeline] Self-improvement failed: {e}")
                    model_comparison = []
                    improvement_history = []
                    best_model_version = None
                    optimization_report = {}
                    validation_results = {}
                finally:
                    improvement_end = time.time()
                    stage_durations["self_improvement"] = improvement_end - improvement_start
                    improvement_summary = {"improvement_history": improvement_history if 'improvement_history' in locals() else [], "model_comparison": model_comparison if 'model_comparison' in locals() else []}
                    if progress_callback:
                        progress_callback({"stage": "self_improvement", "status": "completed", "percent": 80, "result": improvement_summary})
                    self._notify_progress(
                        progress_callback,
                        "hyperparameter_optimization",
                        "completed",
                        78,
                        optimization_report if isinstance(optimization_report, dict) else {},
                    )
                    self._notify_progress(
                        progress_callback,
                        "model_comparison",
                        "completed",
                        82,
                        model_comparison if isinstance(model_comparison, list) else [],
                    )

            except Exception as e:
                _logger.warning(f"[MasterPipeline] Model training failed: {e}")
                deployment_readiness = {"risk_level": "Unknown", "warnings": [str(e)]}
        else:
            _logger.warning("[MasterPipeline] No target column detected; skipping model training.")

        # Initialize model_registry_entry to None; it will be set if register_model succeeds.
        # This prevents UnboundLocalError if registration fails.
        model_registry_entry = None
        try:
            model_registry_entry = self.model_registry.register_model(
            model=None,
            model_name=algorithm,
            dataset_name=dataset_name,
            problem_type=problem_type,
            algorithm=algorithm,
            metrics={
                "test_score": 0.0,
                "train_score": 0.0,
            },
            feature_names=dataset.columns.tolist(),
            hyperparameters=hyperparameter_report.get("recommended_parameters", {}),
            training_time=0.0,
            shap_available=True,
            deployment_status="Not Deployed",
            artifact_path=None,
        )
        except Exception as e:
            _logger.warning(f"[MasterPipeline] Model registry failed: {e}")
            model_registry_entry = None

        # If we created baseline artifacts earlier, attach them to the registry entry
        if model_registry_entry:
            try:
                if 'model_versions' in locals() and model_versions:
                    latest = model_versions[-1]
                    # Overwrite artifact_path if missing or None
                    try:
                        model_registry_entry["artifact_path"] = latest.get("artifact_path")
                    except Exception:
                        model_registry_entry.setdefault("artifact_path", latest.get("artifact_path"))
                    model_registry_entry.setdefault("extra_metadata", {})["encoder_path"] = latest.get("encoder_path")
                    model_registry_entry.setdefault("extra_metadata", {})["preprocessing_path"] = latest.get("preprocessing_path")
                    model_registry_entry.setdefault("extra_metadata", {})["feature_schema"] = latest.get("feature_schema")
                else:
                    # attempt to discover latest artifact on disk
                    try:
                        dp = os.path.join("deployment_package")
                        if os.path.exists(dp):
                            deps = sorted([d for d in os.listdir(dp) if d.startswith("DEP_")])
                            if deps:
                                latest_dep = deps[-1]
                                v1path = os.path.join(dp, latest_dep, "v1", "best_model.pkl")
                                if os.path.exists(v1path):
                                    model_registry_entry["artifact_path"] = v1path
                                    model_registry_entry.setdefault("extra_metadata", {})["encoder_path"] = os.path.join(dp, latest_dep, "v1", "target_encoder.pkl")
                                    model_registry_entry.setdefault("extra_metadata", {})["preprocessing_path"] = os.path.join(dp, latest_dep, "v1", "preprocessing_pipeline.pkl")
                                    model_registry_entry.setdefault("extra_metadata", {})["feature_schema"] = os.path.join(dp, latest_dep, "v1", "feature_schema.json")
                    except Exception:
                        pass
            except Exception:
                pass

        # Collect pipeline execution logs for each major stage (lightweight)
        try:
            pipeline_logs = []
            def _log_stage(name, start_ts, end_ts, input_shape=None, output_keys=None, status="success", note=None):
                pipeline_logs.append({
                    "stage": name,
                    "start_time": start_ts.isoformat() + "Z" if hasattr(start_ts, 'isoformat') else str(start_ts),
                    "end_time": end_ts.isoformat() + "Z" if hasattr(end_ts, 'isoformat') else str(end_ts),
                    "duration_sec": float((end_ts - start_ts).total_seconds()) if hasattr(start_ts, 'isoformat') else None,
                    "input_shape": input_shape,
                    "output_keys": output_keys,
                    "status": status,
                    "note": note,
                })

            # create logs for observed prints/steps
            now = datetime.datetime.utcnow()
            _log_stage("dataset_intelligence", now, now, input_shape=[int(dataset.shape[0]), int(dataset.shape[1])], output_keys=list(dataset_report.keys()) if isinstance(dataset_report, dict) else None)
            _log_stage("cleaning", now, now, input_shape=[int(dataset.shape[0]), int(dataset.shape[1])], output_keys=list(cleaning_results.keys()) if isinstance(cleaning_results, dict) else None)
            _log_stage("eda", now, now, input_shape=[int(cleaned_df.shape[0]), int(cleaned_df.shape[1])], output_keys=list(eda_results.keys()) if isinstance(eda_results, dict) else None)
            _log_stage("feature_engineering", now, now, input_shape=[int(cleaned_df.shape[0]), int(cleaned_df.shape[1])], output_keys=list(feature_plan.keys()) if isinstance(feature_plan, dict) else None)
            _log_stage("model_training", now, now, input_shape=[int(X.shape[0]) if 'X' in locals() else None, int(X.shape[1]) if 'X' in locals() else None], output_keys=list(model_results.keys()) if isinstance(model_results, dict) else None)
            _log_stage("self_improvement", now, now, input_shape=None, output_keys=["model_comparison","improvement_history"]) 
            # write to file
            try:
                import json
                with open("storage/logs/pipeline_execution_log.json", "w", encoding="utf-8") as f:
                    json.dump({"runs_at": datetime.datetime.utcnow().isoformat() + "Z", "stages": pipeline_logs}, f, indent=2)
            except Exception:
                pass
        except Exception:
            pass

        # deployment_readiness computed earlier or default
        try:
            deployment_readiness = deployment_readiness
        except NameError:
            deployment_readiness = self.deployment_agent.analyze_deployment_risk({"metrics": {}, "training_info": {}, "feature_list": dataset.columns.tolist(), "hyperparameters": {}})

        _logger.debug(f"[MasterPipeline] deployment_readiness keys: {list(deployment_readiness.keys()) if isinstance(deployment_readiness, dict) else type(deployment_readiness)}")

        # Final scoring (before documentation — recommendation is required there)
        dataset_score = dataset_report.get("intelligence_score", {}).get("score", 0)
        model_score = model_results.get("model_readiness_score", 0)
        trust_raw = ai_trust_results.get("trust_score", 0)
        # Ensure trust_score is numeric; convert string "Not Evaluated" to 0 for fallback values
        trust_score = float(trust_raw) if isinstance(trust_raw, (int, float)) else 0
        mlops_score = 100 if deployment_readiness.get("risk_level") == "Low" else 60 if deployment_readiness.get("risk_level") == "Medium" else 30
        deployment_readiness["deployment_score"] = mlops_score
        deployment_readiness["readiness_score"] = mlops_score

        # Compute unified trust score once and keep it numeric for all outputs.
        trust_score_value = TrustScoreCalculator.ensure_numeric_trust_score(ai_trust_results.get("trust_score"))
        try:
            trust_score_value = TrustScoreCalculator.calculate_trust_score(
                model_reliability=model_results.get("best_score"),
                dataset_health=dataset_score,
                explainability_available=(
                    isinstance(explainability_results, dict)
                    and self._has_explainability_feature_importance(explainability_results.get("feature_importance"))
                ),
                fairness_score=ai_trust_results.get("fairness_score", 50),
                privacy_score=50.0,
                deployment_readiness=deployment_readiness,
                model_results=model_results,
            )
        except Exception as exc:
            _logger.warning(f"[MasterPipeline] Trust score calculation failed, falling back to raw trust value: {exc}")

        trust_score_value = TrustScoreCalculator.ensure_numeric_trust_score(trust_score_value)
        ai_trust_results["trust_score"] = trust_score_value
        trust_score = trust_score_value
        overall_score = int(round((dataset_score * 0.4 + model_score * 0.3 + trust_score * 0.2 + mlops_score * 0.1)))
        final_score = overall_score
        confidence_value = float(trust_score)
        recommendation = TrustScoreCalculator.recommendation_from_trust_score(trust_score_value)
        deployment_status = TrustScoreCalculator.deployment_status_from_trust_score(trust_score_value)

        _logger.info(f"FINAL TRUST SCORE = {trust_score_value}")

        try:
            documentation = self.documentation_agent.generate_project_documentation(
                {
                    "project_goal": project_goal,
                    "dataset_name": dataset_name,
                    "dataset_report": dataset_report,
                    "cleaning_results": cleaning_results,
                    "eda_results": eda_results,
                    "feature_plan": feature_plan,
                    "feature_engineering_results": feature_plan,
                    "model_results": model_results,
                    "explainability_results": explainability_results,
                    "hyperparameter_report": hyperparameter_report,
                    "ethics_report": ethics_report,
                    "deployment_readiness": deployment_readiness,
                    "registry_entry": model_registry_entry or {},
                    "recommendation": recommendation,
                }
            )
        except Exception as exc:
            self._record_stage_error("documentation", exc, stage_errors)
            documentation = {"status": "documentation_failed", "error": str(exc), "summary": "Documentation generation failed."}

        pipeline_output = {
            "start_time": start_time,
            "dataset_analysis": dataset_report,
            "dataset_report": dataset_report,
            "cleaning_results": cleaning_results,
            "eda_results": eda_results,
            "feature_engineering_results": feature_plan,
            "model_results": model_results,
            "explainability_results": explainability_results,
            "xai_results": explainability_results,
            "ai_trust_results": ai_trust_results,
            "training_artifacts": training_artifacts,
            "model_comparison": model_comparison if 'model_comparison' in locals() else [],
            "improvement_history": improvement_history if 'improvement_history' in locals() else [],
            "best_model_version": best_model_version if 'best_model_version' in locals() else None,
            "optimization_report": optimization_report if 'optimization_report' in locals() else {},
            "validation_results": validation_results if 'validation_results' in locals() else {},
            "model_registry": model_registry_entry or {},
            "model_versions": model_versions if 'model_versions' in locals() else [],
            "production_model": production_model if 'production_model' in locals() else None,
            "validation_history": (validation_results if isinstance(validation_results, list) else [validation_results]) if 'validation_results' in locals() else [],
            "optimization_iterations": (optimization_report.get('iterations') if isinstance(optimization_report, dict) and 'iterations' in optimization_report else (locals().get('optimization_iterations') if 'optimization_iterations' in locals() else 0)),
            "final_ai_confidence_score": confidence_value,
            "mlops_results": {"deployment_score": mlops_score, "risk_level": deployment_readiness.get("risk_level", "Unknown")},
            "final_scores": {"dataset_score": dataset_score, "model_score": model_score, "trust_score": trust_score_value, "mlops_score": mlops_score, "overall_score": overall_score},
            "hyperparameter_report": hyperparameter_report,
            "ethics_report": ethics_report,
            "memory_updates": experiment_entry,
            "deployment_readiness": deployment_readiness,
            "documentation": documentation,
            "model_registry_entry": model_registry_entry or {},
            "final_score": final_score,
            "recommendation": recommendation,
            "project_goal": project_goal,
            "constraints": constraints,
            "dataset_name": dataset_name,
            "stage_durations": stage_durations,
            "stage_errors": stage_errors,
            "total_runtime": float(time.time() - run_start),
        }

        try:
            executive_metrics = create_executive_metrics_object(
                best_model=model_results.get("best_model"),
                model_version=model_registry_entry.get("version") if model_registry_entry else "v1",
                accuracy=model_results.get("best_score"),
                trust_score=trust_score_value,
                risk_level=deployment_readiness.get("risk_level", "Unknown"),
                deployment_status=deployment_status,
                deployment_readiness=deployment_readiness,
                health_score=dataset_score,
                confidence_score=confidence_value,
                final_decision={
                    "risk_level": deployment_readiness.get("risk_level", "Unknown"),
                    "deployment_status": deployment_status,
                    "trust_score": trust_score_value,
                    "confidence_score": confidence_value,
                    "recommendation": recommendation,
                },
                runtime_seconds=float(time.time() - run_start),
                model_results=model_results,
                deployment_info=deployment_readiness,
            )
            pipeline_output["executive_metrics"] = executive_metrics
        except Exception as exc:
            _logger.error("[MasterPipeline] Failed to create executive metrics: %s", exc)
            pipeline_output["executive_metrics"] = {}

        from utils.safe_checks import resolve_canonical_accuracy, format_accuracy_display

        executive_metrics = coalesce_dict(pipeline_output.get("executive_metrics"))
        canonical_accuracy = resolve_canonical_accuracy(
            executive_metrics,
            model_results,
            best_model=model_results.get("best_model"),
        )
        if canonical_accuracy is not None:
            if 0 < canonical_accuracy <= 1:
                executive_metrics["accuracy"] = round(canonical_accuracy * 100, 2)
            else:
                executive_metrics["accuracy"] = round(canonical_accuracy, 2)
        executive_metrics["accuracy_display"] = format_accuracy_display(
            executive_metrics.get("accuracy") or canonical_accuracy,
            problem_type,
        )
        executive_metrics["problem_type"] = problem_type
        pipeline_output["executive_metrics"] = executive_metrics

        # Build and persist a final report payload and PDF report
        try:
            eda_findings = []
            if isinstance(eda_results, dict):
                eda_findings = coalesce_list(safe_dict_get(eda_results, "insights"))
                if not eda_findings:
                    eda_findings = coalesce_list(safe_dict_get(eda_results, "ai_insights"))
            shap_ranking = []
            if isinstance(explainability_results, dict):
                fi_dict = feature_importance_as_dict(safe_dict_get(explainability_results, "feature_importance"))
                shap_ranking = [
                    f"{name}: {value:.4f}"
                    for name, value in sorted(fi_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                ]
            cleaning_actions = []
            if isinstance(cleaning_report, dict):
                cleaning_actions = coalesce_list(safe_dict_get(cleaning_report, "actions"))
                if not cleaning_actions:
                    cleaning_actions = coalesce_list(safe_dict_get(cleaning_report, "steps"))
            detected_issues = detect_data_issues(dataset) if hasattr(dataset, "columns") else []
            health_grade = dataset_report.get("intelligence_score", {}).get("label")
            if not health_grade:
                health_grade = "Excellent" if dataset_score >= 90 else "Good" if dataset_score >= 75 else "Needs improvement" if dataset_score >= 60 else "Critical"
            feature_engineering_steps = []
            if isinstance(feature_plan, dict):
                feature_engineering_steps = normalize_recommendations(safe_dict_get(feature_plan, "steps"))
            from utils.model_ranking import build_best_model_consistency_notes
            from utils.pdf_visualizations import (
                chart_prefix_from_name,
                ensure_eda_chart_paths,
                ensure_explainability_chart_paths,
            )

            selection_notes = build_best_model_consistency_notes(
                model_results.get("best_model"),
                coalesce_dict(model_results.get("detailed_metrics")),
                problem_type,
                coalesce_dict(model_results.get("metrics")),
                coalesce_dict(model_results.get("model_selection_explanation")),
            )
            dataset_identification = coalesce_dict(dataset_report.get("dataset_identification"))
            chart_prefix = chart_prefix_from_name(dataset_name)
            eda_chart_paths = ensure_eda_chart_paths(
                cleaned_df if isinstance(cleaned_df, pd.DataFrame) else None,
                target,
                eda_results if isinstance(eda_results, dict) else None,
                chart_prefix,
            )
            explainability_chart_paths = ensure_explainability_chart_paths(
                explainability_results if isinstance(explainability_results, dict) else None,
                chart_prefix,
            )
            final_report_payload = {
                "project_name": project_goal,
                "project_goal": project_goal,
                "executive_summary": documentation.get("summary", f"AutoDS completed an end-to-end analysis for {dataset_name}."),
                "dataset_name": dataset_name,
                "uploaded_file": dataset_name,
                "detected_dataset": dataset_identification.get("detected_dataset"),
                "model_selection_notes": selection_notes,
                "rows": int(dataset.shape[0]),
                "columns": int(dataset.shape[1]),
                "health_score": dataset_score,
                "health_grade": health_grade,
                "health_summary": dataset_report.get("problem_analysis", {}).get("summary") or dataset_report.get("summary", "") or f"Overall pipeline score {final_score}/100 with dataset quality {dataset_score}/100.",
                "dataset_summary": dataset_report.get("problem_analysis", {}).get("summary") or dataset_report.get("summary", ""),
                "feature_types": dataset_report.get("feature_types") or {},
                "problem_type": problem_type,
                "best_model": model_results.get("best_model"),
                "accuracy": executive_metrics.get("accuracy"),
                "accuracy_display": executive_metrics.get("accuracy_display"),
                "model_version": (model_registry_entry.get("version") if model_registry_entry else None) or (best_model_version if 'best_model_version' in locals() and best_model_version else "v1"),
                "production_model": production_model if 'production_model' in locals() and production_model else None,
                "model_selection_rationale": documentation.get("sections", {}).get("Model Selection", ""),
                "target_column": target if target is not None else "Auto-detected",
                "confidence": confidence_value,
                "ai_confidence_score": confidence_value,
                "trust_score": trust_score_value,
                "fairness_score": ai_trust_results.get("fairness_score", 50.0),
                "deployment_label": deployment_readiness.get("deployment_label") or deployment_readiness.get("status") or recommendation,
                "final_score": overall_score,
                "feature_importance": explainability_results.get("feature_importance") if isinstance(explainability_results, dict) else {},
                "explainability_summary": (explainability_results.get("explanation") or explainability_results.get("summary") if isinstance(explainability_results, dict) else "") or "",
                "trust_concerns": normalize_recommendations(ai_trust_results.get("bias_concerns")),
                "model_results": [f"{name}: {value:.4f}" for name, value in (model_results.get("metrics") or {}).items()],
                "detailed_metrics": coalesce_dict(model_results.get("detailed_metrics")),
                "training_times": coalesce_dict(model_results.get("training_times")),
                "deployment_status": executive_metrics.get("deployment_status") or deployment_readiness.get("deployment_label") or deployment_readiness.get("status"),
                "feature_engineering_summary": feature_engineering_steps,
                "model_training_summary": [
                    f"Best model: {model_results.get('best_model')}",
                    f"Models evaluated: {len(model_results.get('metrics') or {})}",
                ],
                "ai_insights": list(eda_findings[:8]) if eda_findings else [documentation.get("summary", "")],
                "eda_findings": list(eda_findings[:12]) if isinstance(eda_findings, list) else [],
                "eda_chart_paths": eda_chart_paths,
                "explainability_chart_paths": explainability_chart_paths,
                "cleaning_actions": cleaning_actions,
                "cleaning_before_rows": int(dataset.shape[0]),
                "cleaning_after_rows": int(cleaned_df.shape[0]) if isinstance(cleaned_df, pd.DataFrame) else int(dataset.shape[0]),
                "data_quality_assessment": cleaning_actions or ["Data quality checks and automated cleaning were completed."],
                "ai_recommendations": normalize_recommendations(recommendation) + normalize_recommendations(documentation.get("sections", {}).get("Deployment Recommendation", "")) + normalize_recommendations(ai_trust_results.get("bias_concerns")),
                "recommendations": [
                    recommendation,
                    documentation.get("sections", {}).get("Deployment Recommendation", ""),
                ],
                "issues": detected_issues[:8],
                "shap_ranking": shap_ranking,
                "deployment_readiness_score": mlops_score,
                "deployment_risk": deployment_readiness.get("risk_level", "Unknown"),
                "business_recommendations": documentation.get("sections", {}).get("Future Improvements", ""),
                "health_breakdown": {
                    "dataset": dataset_score,
                    "model": model_score,
                    "trust": trust_score,
                    "mlops": mlops_score,
                },
                "final_conclusion": recommendation or documentation.get("sections", {}).get("Future Improvements", ""),
                "conclusion": recommendation,
                "generated_at": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            }
            try:
                report_path = generate_pdf_report(final_report_payload)
            except Exception as exc:
                self._record_stage_error("final_report_generation", exc, stage_errors)
                report_path = None
            pipeline_output["final_report"] = {"path": report_path, "payload": final_report_payload}
            self._notify_progress(
                progress_callback,
                "pdf_report",
                "completed",
                95,
                pipeline_output["final_report"],
            )
        except Exception as exc:
            self._record_stage_error("final_report", exc, stage_errors)
            pipeline_output["final_report"] = {"path": None, "payload": {}, "error": str(exc)}

        self._notify_progress(
            progress_callback,
            "ai_decision",
            "completed",
            98,
            {
                "recommendation": recommendation,
                "final_score": final_score,
                "best_model": model_results.get("best_model"),
            },
        )

        if smart_mode:
            try:
                cache_output = dict(pipeline_output)
                artifacts_cache = cache_output.get("training_artifacts")
                if isinstance(artifacts_cache, dict):
                    cache_output["training_artifacts"] = {
                        key: value
                        for key, value in artifacts_cache.items()
                        if key not in {"best_model", "target_encoder", "X_data", "y_data", "cleaned_dataframe", "extras"}
                    }
                with open(cache_path, 'w', encoding='utf-8') as fh:
                    json.dump(cache_output, fh, default=str, indent=2)
            except Exception:
                pass

        try:
            if stage_durations:
                slowest_stage, slowest_duration = max(stage_durations.items(), key=lambda kv: kv[1])
                _logger.info(f"[MasterPipeline] Slowest stage: {slowest_stage} took {slowest_duration:.2f}s")
                pipeline_output["slowest_stage"] = {
                    "stage": slowest_stage,
                    "duration_sec": round(slowest_duration, 3),
                }
        except Exception:
            pass

        self.agent_memory_db.learn_from_project(
            {
                "agent": "MasterAutonomousPipeline",
                "title": f"Pipeline run for {dataset_name}",
                "outcome": recommendation,
                "performance": final_score,
                "tags": [dataset_name, "pipeline", model_strategy.get("problem_type", "Unknown")],
                "importance": 0.9,
                "success_score": float(final_score / 100.0),
            }
        )

        # Ensure a persistent pipeline execution log is written based on pipeline_output
        try:
            import json
            log_stages = []
            now_ts = datetime.datetime.utcnow().isoformat() + "Z"
            # Define canonical stages and map to pipeline keys
            stage_map = [
                ("dataset_intelligence", "dataset_analysis"),
                ("cleaning", "cleaning_results"),
                ("eda", "eda_results"),
                ("feature_engineering", "feature_engineering_results"),
                ("model_training", "model_results"),
                ("self_improvement", "improvement_history"),
                ("model_comparison", "model_comparison"),
                ("explainability", "explainability_results"),
                ("ai_trust", "ai_trust_results"),
                ("deployment_readiness", "deployment_readiness"),
                ("documentation", "documentation"),
            ]
            for name, key in stage_map:
                obj = pipeline_output.get(key)
                output_keys = list(obj.keys()) if isinstance(obj, dict) else None
                input_shape = None
                if key == 'dataset_analysis' and isinstance(pipeline_output.get('dataset_analysis'), dict):
                    ds = pipeline_output.get('dataset_analysis')
                    shape = ds.get('dataset_shape')
                    if isinstance(shape, dict):
                        input_shape = [shape.get('rows'), shape.get('columns')]
                log_stages.append({
                    "stage": name,
                    "start_time": pipeline_output.get('start_time'),
                    "end_time": now_ts,
                    "duration_sec": None,
                    "input_shape": input_shape,
                    "output_keys": output_keys,
                    "status": "success" if obj is not None else "missing",
                })
            with open("storage/logs/pipeline_execution_log.json", "w", encoding="utf-8") as f:
                json.dump({"run_at": pipeline_output.get('start_time'), "generated_at": now_ts, "stages": log_stages}, f, indent=2)
        except Exception:
            pass

        return pipeline_output
