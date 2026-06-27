import json
import os
import time
import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from agents.natural_language_control_agent import NaturalLanguageControlAgent
from agents.dataset_intelligence_agent import DatasetIntelligenceAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.hyperparameter_agent import HyperparameterOptimizationAgent
from agents.experiment_memory_agent import ExperimentMemoryAgent
from agents.model_registry_agent import ModelRegistryAgent
from agents.api_deployment_agent import APIDeploymentAgent
from agents.drift_monitoring_agent import DataDriftMonitoringAgent
from agents.autonomous_data_scientist_agent import AutonomousDataScientistAgent
from agents.self_healing_agent import SelfHealingAgent


class AgentExecutionEngine:
    DEFAULT_HISTORY = "agent_execution_history.json"
    MAX_HISTORY = 5000

    CONFIDENCE_WEIGHTS = {
        "dataset": 20,
        "feature": 20,
        "model_optimization": 30,
        "deployment": 30,
    }

    def __init__(self, history_path: Optional[str] = None) -> None:
        self.history_path = history_path or os.path.join(os.getcwd(), self.DEFAULT_HISTORY)
        self.history: List[Dict[str, Any]] = self._load_history()
        self.nl_agent = NaturalLanguageControlAgent()
        self.self_healing = SelfHealingAgent()

    def _load_history(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.history_path):
            return []
        try:
            with open(self.history_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
        except (OSError, json.JSONDecodeError):
            pass
        return []

    def _persist_history(self) -> None:
        directory = os.path.dirname(self.history_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(self.history_path, "w", encoding="utf-8") as fh:
            json.dump(self.history[-self.MAX_HISTORY :], fh, indent=2)

    def execute_command(self, command: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        understood = self.nl_agent.understand_command(command)
        intent = understood.get("intent", "unknown")
        entities = understood.get("entities", {})

        plan = self.nl_agent.create_execution_plan(intent, entities)

        start_ts = datetime.datetime.utcnow().isoformat() + "Z"
        pipeline_results = self.run_pipeline(plan, context)

        # compute confidence
        score = 0
        overall_success = all(item.get("status") == "SUCCESS" for item in pipeline_results)
        for item in pipeline_results:
            act = item.get("step", "").lower()
            if "feature" in act:
                score += self.CONFIDENCE_WEIGHTS["feature"] if item.get("status") == "SUCCESS" else 0
            if "dataset" in act or "analyze" in act:
                score += self.CONFIDENCE_WEIGHTS["dataset"] if item.get("status") == "SUCCESS" else 0
            if "hyper" in act or "optimi" in act or "model" in act:
                score += self.CONFIDENCE_WEIGHTS["model_optimization"] if item.get("status") == "SUCCESS" else 0
            if "deploy" in act:
                score += self.CONFIDENCE_WEIGHTS["deployment"] if item.get("status") == "SUCCESS" else 0

        confidence = min(1.0, score / 100.0)

        # human-friendly explanation
        explanation = self._generate_explanation(command, intent, pipeline_results)

        entry = {
            "timestamp": start_ts,
            "command": command,
            "intent": intent,
            "plan": plan,
            "pipeline": pipeline_results,
            "overall_success": overall_success,
            "confidence": confidence,
        }
        self.history.append(entry)
        self.history = self.history[-self.MAX_HISTORY :]
        self._persist_history()

        return {
            "intent": intent,
            "plan": plan,
            "pipeline": pipeline_results,
            "overall_success": overall_success,
            "confidence": confidence,
            "explanation": explanation,
        }

    def run_pipeline(self, steps: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        context = context or {}
        results: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps, start=1):
            action = step.get("action") or step.get("step") or str(step)
            name = action if isinstance(action, str) else str(action)
            t0 = time.time()
            try:
                result = self._execute_action(name, context)
                status = "SUCCESS"
                summary = result.get("summary") if isinstance(result, dict) else str(result)
            except Exception as exc:
                status = "FAILED"
                summary = str(exc)
                # analyze and recommend fix
                analysis = self.self_healing.analyze_error(str(exc), {})
                recovery = self.self_healing.recommend_fix(analysis)
                result = {"error": str(exc), "analysis": analysis, "recovery": recovery}
                # attempt automated retry for safe fixes
                if recovery.get("automation_possible"):
                    try:
                        repaired_context = self._attempt_auto_fix(context, analysis)
                        retry_result = self._execute_action(name, repaired_context)
                        status = "SUCCESS"
                        summary = retry_result.get("summary") if isinstance(retry_result, dict) else str(retry_result)
                        result = {"recovery_attempt": True, "result": retry_result}
                    except Exception as exc2:
                        status = "FAILED"
                        result["retry_error"] = str(exc2)

            t1 = time.time()
            duration = round(t1 - t0, 3)
            results.append({
                "step": name,
                "status": status,
                "execution_time": duration,
                "result_summary": summary,
                "result": result,
            })

        return results

    def _execute_action(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        a = action.lower()
        if "dataset" in a or "analy" in a:
            df = context.get("dataset")
            if df is None or not isinstance(df, pd.DataFrame):
                raise ValueError("No dataset provided for analysis")
            agent = DatasetIntelligenceAgent(df, name=context.get("dataset_name", "dataset"))
            domain = agent.detect_domain()
            problem = agent.analyze_problem()
            risks = agent.assess_risks()
            return {"summary": f"Domain {domain.get('domain')}, problem {problem.get('problem_type')}", "domain": domain, "problem": problem, "risks": risks}

        if "feature" in a:
            df = context.get("dataset")
            if df is None or not isinstance(df, pd.DataFrame):
                raise ValueError("No dataset provided for feature engineering")
            agent = FeatureEngineeringAgent(df, name=context.get("dataset_name", "dataset"))
            plan = agent.generate_feature_plan()
            suggestions = agent.suggest_new_features()
            return {"summary": f"Recommended {plan.get('recommended_changes')} feature changes", "plan": plan, "suggestions": suggestions}

        if "hyper" in a or "optimi" in a or ("train" in a and "model" in a):
            # Expect X and y in context or dataset
            X, y = self._extract_xy(context)
            if X is None or y is None:
                raise ValueError("No X/y available for hyperparameter optimization")
            agent = HyperparameterOptimizationAgent()
            # choose a simple model name from registry
            model_candidates = agent.supported_models("Classification")
            model_name = model_candidates[0] if model_candidates else list(agent.PARAM_GRID.keys())[0]
            result = agent.optimize(X, y, model_name, use_history=False)
            return {"summary": result.get("status"), "details": result}

        if "register" in a or "register_model" in a or "register" in a:
            registry = ModelRegistryAgent()
            # minimal registration using placeholders
            model = context.get("model")
            if model is None:
                model = None
            metrics = context.get("metrics", {})
            entry = registry.register_model(model, model_name=context.get("model_name", "model"), dataset_name=context.get("dataset_name", "dataset"), problem_type=context.get("problem_type", "Unknown"), algorithm=context.get("algorithm", "Unknown"), metrics=metrics, feature_names=context.get("feature_names", []), hyperparameters=context.get("hyperparameters", {}), training_time=context.get("training_time", 0.0))
            return {"summary": f"Registered model {entry.get('model_id')}", "entry": entry}

        if "deploy" in a:
            deployer = APIDeploymentAgent()
            model_path = context.get("model_path")
            feature_names = context.get("feature_names", [])
            if not model_path or not os.path.exists(model_path):
                return {"summary": "No model artifact provided; skipped deployment", "status": "skipped"}
            pkg = deployer.build_deployment_package(model_path, feature_names)
            return {"summary": "Deployment package built", "package": pkg}

        if "drift" in a or "monitor" in a:
            reference = context.get("reference")
            new_data = context.get("new_data")
            if reference is None or new_data is None:
                raise ValueError("Reference data and new_data required for drift detection")
            agent = DataDriftMonitoringAgent()
            ref = agent.load_reference_data() if os.path.exists(agent.reference_path) else reference
            res = agent.detect_drift(ref, new_data)
            return {"summary": f"Drift severity {res.get('severity')}", "details": res}

        if "autonomous" in a:
            dataset = context.get("dataset")
            goal = context.get("goal", "Auto project")
            agent = AutonomousDataScientistAgent()
            res = agent.run_autonomous_project(dataset, goal)
            return {"summary": "Autonomous run completed", "details": res}

        # default: route to chief
        chief = NaturalLanguageControlAgent()
        routed = chief.route_command(a)
        return {"summary": f"Routed to {routed}"}

    def _extract_xy(self, context: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        if "X" in context and "y" in context:
            return context["X"], context["y"]
        df = context.get("dataset")
        if isinstance(df, pd.DataFrame) and df.shape[1] >= 2:
            # assume last column is target
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            return X, y
        return None, None

    def _attempt_auto_fix(self, context: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        # Perform conservative automated fixes for common issues
        error_type = analysis.get("error_type", "")
        ctx = dict(context)
        df = ctx.get("dataset")
        if df is None or not isinstance(df, pd.DataFrame):
            return ctx

        if error_type == "Encoding Error":
            obj_cols = [c for c in df.columns if df[c].dtype == object]
            if obj_cols:
                df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
                ctx["dataset"] = df
        if error_type == "Missing Value Error":
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")
            ctx["dataset"] = df

        return ctx

    def _generate_explanation(self, command: str, intent: str, pipeline_results: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for item in pipeline_results:
            step = item.get("step", "")
            status = item.get("status")
            if status == "SUCCESS":
                parts.append(f"{step} succeeded")
            else:
                parts.append(f"{step} failed")

        return f"Executed '{command}'. " + ", ".join(parts)
