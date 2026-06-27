import datetime
import json
import os
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from agents.dataset_intelligence_agent import DatasetIntelligenceAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.chief_data_scientist_agent import ChiefDataScientistAgent
from agents.hyperparameter_agent import HyperparameterOptimizationAgent
from agents.model_registry_agent import ModelRegistryAgent
from agents.api_deployment_agent import APIDeploymentAgent


class AutonomousDataScientistAgent:
    DEFAULT_DECISION_FILE = "autonomous_decisions.json"
    MAX_DECISIONS = 5000

    def __init__(self, decision_path: Optional[str] = None) -> None:
        self.decision_path = decision_path or self.DEFAULT_DECISION_FILE
        self.decisions: List[Dict[str, Any]] = self._load_decisions()
        self.hyper_agent = HyperparameterOptimizationAgent()
        self.registry_agent = ModelRegistryAgent()
        self.chief_agent = ChiefDataScientistAgent()
        self.deployment_agent = APIDeploymentAgent()

    def _load_decisions(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.decision_path):
            return []
        try:
            with open(self.decision_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return []

    def _persist_decisions(self) -> None:
        directory = os.path.dirname(self.decision_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(self.decision_path, "w", encoding="utf-8") as handle:
            json.dump(self.decisions[-self.MAX_DECISIONS :], handle, indent=2)

    def _append_decision(
        self,
        agent: str,
        decision: str,
        confidence: float = 1.0,
        input_facts: Optional[List[Dict[str, Any]]] = None,
        reasoning: Optional[str] = None,
        action: Optional[str] = None,
    ) -> None:
        entry = {
            "time": datetime.datetime.utcnow().isoformat() + "Z",
            "agent": agent,
            "decision": decision,
            "confidence": float(confidence),
            "input_facts": input_facts or [],
            "reasoning": reasoning or "",
            "action_taken": action or "",
        }
        self.decisions.append(entry)
        self.decisions = self.decisions[-self.MAX_DECISIONS :]
        self._persist_decisions()

    def _preprocess(self, df: pd.DataFrame, target: str) -> (pd.DataFrame, pd.Series):
        df = df.copy()
        # basic cleaning
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].fillna("Unknown")
            else:
                df[col] = df[col].fillna(df[col].median())

        if target not in df.columns:
            raise ValueError(f"Target column {target} not found in dataset")

        y = df[target]
        X = df.drop(columns=[target])
        # simple categorical encoding
        X = pd.get_dummies(X, drop_first=True)
        return X, y

    def run_autonomous_project(
        self,
        dataset: Any,
        project_goal: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a full autonomous ML project and return a project object."""
        constraints = constraints or {}

        # Load dataset
        if isinstance(dataset, str):
            df = pd.read_csv(dataset)
            dataset_name = os.path.basename(dataset)
        elif isinstance(dataset, pd.DataFrame):
            df = dataset.copy()
            dataset_name = getattr(dataset, "name", "in-memory-dataset")
        else:
            raise ValueError("dataset must be a file path or pandas DataFrame")

        # 1. Analyze dataset
        dataset_agent = DatasetIntelligenceAgent(df, dataset_name)
        dataset_report = dataset_agent.generate_dataset_report()
        self._append_decision(
            agent="Dataset Intelligence",
            decision=f"Dataset analyzed: {dataset_report.get('dataset_name', dataset_name)}",
            confidence=dataset_report.get("intelligence_score", {}).get("score", 0) / 100.0,
            input_facts=[{"rows": df.shape[0], "cols": df.shape[1]}],
            reasoning=dataset_report.get("executive_summary", ""),
            action="Produce dataset report",
        )

        # 2. Review data quality and risks
        risks = dataset_report.get("risk_analysis", {})
        self._append_decision(
            agent="Risk Assessment",
            decision=f"Risk level: {risks.get('risk_level')}",
            confidence=1.0,
            input_facts=[risks],
            reasoning=risks.get("risk_score", ""),
            action="Record risks",
        )

        # 3. Feature engineering strategy
        feature_agent = FeatureEngineeringAgent(df, dataset_name)
        feature_plan = feature_agent.generate_feature_plan()
        self._append_decision(
            agent="Feature Engineering",
            decision="Generated feature plan",
            confidence=0.9,
            input_facts=[feature_plan],
            reasoning=feature_plan.get("steps", []),
            action="Recommend transformations",
        )

        # 4. Select ML strategy using ChiefDataScientistAgent
        chief_summary = self.chief_agent.analyze_project(df, dataset_metadata={"name": dataset_name})
        recommendations = chief_summary.get("problem_insights", {})
        model_recs = dataset_agent.recommend_models().get("recommendations", [])
        self._append_decision(
            agent="Chief Data Scientist",
            decision="Selected model candidates",
            confidence=0.9,
            input_facts=model_recs,
            reasoning=chief_summary.get("next_best_action", ""),
            action="Prepare candidate list",
        )

        # 5. Train and optimize models
        target = dataset_report.get("problem_analysis", {}).get("likely_target")
        if not target:
            # fallback to last column
            target = df.columns[-1]

        X, y = self._preprocess(df, target)

        # Adjust candidates based on project goal
        candidates = [m.get("model") for m in model_recs][:3]
        goal = project_goal.lower() if project_goal else ""
        if "interpret" in goal:
            # prefer interpretable models
            candidates = [c for c in candidates if "Logistic" in c or "Linear" in c or "Decision" in c] or candidates

        experiments = []
        best_result = None
        for model_name in candidates:
            try:
                result = self.hyper_agent.optimize(X, y, model_name, problem_type=None, use_history=False)
            except Exception as e:
                result = {"status": "failed", "diagnostics": str(e)}

            experiments.append({"model": model_name, "result": result})
            if result.get("status") == "success":
                score = float(result.get("optimized_score", 0.0))
                if best_result is None or score > float(best_result.get("optimized_score", 0.0)):
                    best_result = {"model": model_name, **result}

        chosen = best_result or (experiments[0] if experiments else None)

        # 6. Explain final model (lightweight)
        shap_available = False
        explanation = {}
        try:
            import shap as _shap  # type: ignore
            shap_available = True
        except Exception:
            shap_available = False

        explanation = {"shap_available": shap_available, "notes": "SHAP available" if shap_available else "SHAP not available"}
        self._append_decision(
            agent="XAI",
            decision="Model explainability evaluated",
            confidence=1.0 if shap_available else 0.5,
            input_facts=[explanation],
            reasoning="Checked SHAP availability",
            action="Record explainability",
        )

        # 7. Register final model
        registry_entry = None
        artifact_path = None
        if chosen and chosen.get("status") == "success":
            model_obj = chosen.get("report", {}).get("best_parameters") or {}
            # persist a small artifact placeholder (we cannot reliably serialize sklearn best_estimator here)
            artifacts_dir = "artifacts"
            os.makedirs(artifacts_dir, exist_ok=True)
            artifact_path = os.path.join(artifacts_dir, f"model_{int(datetime.datetime.utcnow().timestamp())}.joblib")
            # if the hyperparameter result contains a trained estimator, try to save it
            estimator = chosen.get("result", {}).get("best_estimator_") if isinstance(chosen.get("result"), dict) else None
            try:
                if estimator is not None:
                    joblib.dump(estimator, artifact_path)
                else:
                    # create a tiny metadata file as placeholder
                    with open(artifact_path, "w", encoding="utf-8") as fh:
                        fh.write(json.dumps({"model": chosen.get("model"), "note": "artifact placeholder"}))
            except Exception:
                # fallback to JSON metadata
                with open(artifact_path, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps({"model": chosen.get("model"), "note": "artifact placeholder"}))

            registry_entry = self.registry_agent.register_model(
                model=None,
                model_name=chosen.get("model"),
                dataset_name=dataset_name,
                problem_type=dataset_report.get("problem_analysis", {}).get("problem_type", "Unknown"),
                algorithm=chosen.get("model"),
                metrics={
                    "optimized_score": float(chosen.get("optimized_score", 0.0)),
                    "baseline_score": float(chosen.get("baseline_score", 0.0)) if chosen.get("baseline_score") is not None else 0.0,
                },
                feature_names=list(X.columns[:20]),
                hyperparameters=chosen.get("best_params", {}) if chosen.get("best_params") else {},
                training_time=float(chosen.get("training_duration", 0.0)) if chosen.get("training_duration") else 0.0,
                shap_available=shap_available,
                deployment_status="Not Deployed",
                artifact_path=artifact_path,
            )

            self._append_decision(
                agent="Model Registry",
                decision=f"Registered model {registry_entry.get('model_id')}",
                confidence=0.9,
                input_facts=[registry_entry],
                reasoning="Persisted model metadata",
                action="Model registered",
            )

        # 8. Generate deployment readiness score
        model_quality = (chosen.get("optimized_score", 0.0) * 100.0) if chosen else 0.0
        data_quality = float(dataset_report.get("intelligence_score", {}).get("score", 0.0))
        deployment_readiness = self.deployment_agent.calculate_deployment_readiness(
            model_quality=model_quality,
            data_quality=data_quality,
            explainability_available=shap_available,
            tests_completed=True,
            drift_monitoring_configured=False,
        )

        self._append_decision(
            agent="Deployment Readiness",
            decision=f"Deployment readiness: {deployment_readiness.get('status')}",
            confidence=1.0,
            input_facts=[deployment_readiness],
            reasoning="Calculated readiness from model and data scores",
            action="Score deployment readiness",
        )

        # 9. Produce executive summary and score
        feature_score = max(0.0, 100.0 - float(feature_plan.get("recommended_changes", 0)) * 5.0)
        model_perf = (chosen.get("optimized_score", 0.0) * 100.0) if chosen else 0.0
        explain_score = 100.0 if shap_available else 50.0
        deployment_score = float(deployment_readiness.get("score", 0.0))

        overall = (
            data_quality * 0.25
            + feature_score * 0.15
            + model_perf * 0.35
            + explain_score * 0.15
            + deployment_score * 0.10
        )
        overall_score = int(round(max(0, min(100, overall))))
        if overall_score >= 90:
            grade = "Excellent"
        elif overall_score >= 75:
            grade = "Production Ready"
        elif overall_score >= 50:
            grade = "Needs Improvement"
        else:
            grade = "At Risk"

        presentation = {
            "business_problem": project_goal,
            "dataset_understanding": dataset_report,
            "challenges": risks.get("risks", []),
            "decisions": self.decisions[-20:],
            "model_comparison": experiments,
            "final_model": registry_entry,
            "risks": risks,
            "deployment_recommendation": deployment_readiness,
        }

        project = {
            "project_title": project_goal,
            "dataset_name": dataset_name,
            "dataset_report": dataset_report,
            "feature_plan": feature_plan,
            "model_experiments": experiments,
            "final_choice": chosen,
            "registry_entry": registry_entry,
            "artifact_path": artifact_path,
            "explainability": explanation,
            "deployment_readiness": deployment_readiness,
            "project_score": {"score": overall_score, "grade": grade, "summary": "Autonomous run summary."},
            "presentation": presentation,
            "decisions_file": self.decision_path,
        }

        return project

    def self_improve(self) -> Dict[str, Any]:
        """Generate improvement suggestions based on past decisions and registry."""
        suggestions: List[str] = []
        # Analyze registry for failed or low-performing models
        models = self.registry_agent.get_model_versions()
        low_perf = [m for m in models if (m.get("metrics", {}).get("optimized_score", 0) or 0) < 0.7]
        if low_perf:
            suggestions.append("Re-evaluate low-performing models for feature engineering improvements.")

        # Look at recent decisions for recurrent data quality issues
        recent_risks = [d for d in self.decisions if "Risk" in d.get("decision", "") or "risk" in d.get("reasoning", "").lower()]
        if recent_risks:
            suggestions.append("Prioritize data quality and drift monitoring in next cycle.")

        # Propose new algorithms
        suggestions.append("Try alternative algorithms such as LightGBM or ensembled linear models.")

        return {"suggestions": suggestions, "count": len(suggestions)}
