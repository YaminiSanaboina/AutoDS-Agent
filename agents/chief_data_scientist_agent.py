"""Chief Data Scientist Agent for AutoDS Agent.

This backend-only module coordinates dataset analysis, modeling, deployment,
and monitoring agents to generate high-level AI recommendations and decisions.
"""
from __future__ import annotations

import datetime
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agents.dataset_intelligence_agent import DatasetIntelligenceAgent
from agents.deployment_agent import DeploymentAgent
from agents.experiment_memory_agent import ExperimentMemoryAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.hyperparameter_agent import HyperparameterOptimizationAgent
from agents.self_healing_agent import SelfHealingAgent
from agents.drift_monitoring_agent import DataDriftMonitoringAgent
from agents.retraining_agent import AutonomousRetrainingAgent
from agents.supervisor_agent import AISupervisor
from agents.workflow_agent import AIWorkflowAgent


class ChiefDataScientistAgent:
    DEFAULT_DECISION_FILE = "storage/monitoring/chief_decisions.json"
    MAX_DECISIONS = 1000

    def __init__(self, decision_path: Optional[str] = None) -> None:
        self.decision_path = decision_path or self.DEFAULT_DECISION_FILE
        self.experiment_agent = ExperimentMemoryAgent()
        self.dataset_agent: Optional[DatasetIntelligenceAgent] = None
        self.feature_agent: Optional[FeatureEngineeringAgent] = None
        self.hyperparameter_agent = HyperparameterOptimizationAgent()
        self.self_healing_agent = SelfHealingAgent()
        self.drift_agent = DataDriftMonitoringAgent()
        self.retraining_agent = AutonomousRetrainingAgent()
        self.deployment_agent = DeploymentAgent()
        self.workflow_agent = self._load_workflow_agent()
        self.supervisor = self._load_supervisor()
        self.decisions = self._load_decisions()

    def _load_workflow_agent(self) -> Optional[AIWorkflowAgent]:
        try:
            return AIWorkflowAgent()
        except Exception:
            return None

    def _load_supervisor(self) -> Optional[AISupervisor]:
        try:
            return AISupervisor()
        except Exception:
            return None

    def _load_decisions(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.decision_path):
            return []
        try:
            with open(self.decision_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
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
        project_stage: str,
        decision: str,
        reasoning: str,
        agents_involved: List[str],
        expected_outcome: str,
    ) -> None:
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "project_stage": project_stage,
            "decision": decision,
            "reasoning": reasoning,
            "agents_involved": agents_involved,
            "expected_outcome": expected_outcome,
        }
        self.decisions.append(entry)
        self.decisions = self.decisions[-self.MAX_DECISIONS :]
        self._persist_decisions()

    def analyze_project(
        self,
        dataset: pd.DataFrame,
        dataset_metadata: Optional[Dict[str, Any]] = None,
        model_history: Optional[List[Dict[str, Any]]] = None,
        drift_history: Optional[List[Dict[str, Any]]] = None,
        deployment_status: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        dataset_metadata = dataset_metadata or {}
        model_history = model_history or []
        drift_history = drift_history or []
        deployment_status = deployment_status or {}

        self.dataset_agent = DatasetIntelligenceAgent(dataset, dataset_metadata.get("name", "Untitled Dataset"))
        dataset_report = self.dataset_agent.generate_dataset_report()
        problem_insights = self.dataset_agent.analyze_problem()
        risk_assessment = self.dataset_agent.assess_risks()
        roadmap = self.dataset_agent.generate_roadmap()

        workflow_status = {}
        if self.workflow_agent is not None:
            try:
                workflow_status = self.workflow_agent.analyze_workflow()
            except Exception:
                workflow_status = {}

        stage = self._infer_project_stage(
            dataset,
            dataset_metadata,
            model_history,
            drift_history,
            deployment_status,
            workflow_status,
        )

        health = self.calculate_project_health(
            dataset_metadata=dataset_metadata,
            model_history=model_history,
            drift_history=drift_history,
            deployment_status=deployment_status,
        )

        strengths = self._summarize_strengths(dataset_report, model_history, deployment_status)
        weaknesses = self._summarize_weaknesses(risk_assessment, deployment_status)
        risks = self._summarize_risks(risk_assessment, drift_history)
        next_action = self.give_strategic_advice(
            dataset_report=dataset_report,
            model_history=model_history,
            drift_history=drift_history,
            deployment_status=deployment_status,
            risk_assessment=risk_assessment,
        )

        decision_summary = {
            "project_stage": stage,
            "project_health": health["score"],
            "health_label": health["label"],
            "strengths": strengths,
            "weaknesses": weaknesses,
            "risks": risks,
            "next_best_action": next_action[0]["recommendation"] if next_action else "Continue monitoring.",
            "dataset_report": dataset_report,
            "problem_insights": problem_insights,
            "risk_assessment": risk_assessment,
            "roadmap": roadmap,
            "workflow_status": workflow_status,
        }

        self._append_decision(
            project_stage=stage,
            decision=decision_summary["next_best_action"],
            reasoning="Aggregated dataset, model, drift, and deployment signals.",
            agents_involved=[
                "DatasetIntelligenceAgent",
                "ExperimentMemoryAgent",
                "AIWorkflowAgent",
                "DataDriftMonitoringAgent",
                "AutonomousRetrainingAgent",
                "DeploymentAgent",
            ],
            expected_outcome="Align the project to the next best workflow step and reduce risk.",
        )

        return decision_summary

    def give_strategic_advice(
        self,
        dataset_report: Optional[Dict[str, Any]] = None,
        model_history: Optional[List[Dict[str, Any]]] = None,
        drift_history: Optional[List[Dict[str, Any]]] = None,
        deployment_status: Optional[Dict[str, Any]] = None,
        risk_assessment: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        dataset_report = dataset_report or {}
        model_history = model_history or []
        drift_history = drift_history or []
        deployment_status = deployment_status or {}
        risk_assessment = risk_assessment or {}

        recommendations: List[Dict[str, Any]] = []
        used = set()

        missing = risk_assessment.get("missing_values", 0.0)
        duplicates = risk_assessment.get("duplicates", 0.0)
        if missing >= 2.0 or duplicates >= 1.0:
            recommendations.append({
                "recommendation": "Improve data quality through cleaning and imputation.",
                "priority": "Critical",
            })
            used.add("cleaning")

        dataset_size_warning = any("small dataset" in r.lower() for r in dataset_report.get("risks", [])) if isinstance(dataset_report.get("risks"), list) else False
        if dataset_size_warning:
            recommendations.append({
                "recommendation": "Add more data to reduce overfitting risk.",
                "priority": "High",
            })
            used.add("data")

        if drift_history and any(entry.get("severity") == "High" for entry in drift_history):
            recommendations.append({
                "recommendation": "Retrain the model to address critical data drift.",
                "priority": "High",
            })
            used.add("retrain")

        if model_history:
            latest = model_history[0]
            train = latest.get("train_score")
            test = latest.get("test_score")
            if train is not None and test is not None and train - test >= 0.15:
                recommendations.append({
                    "recommendation": "Optimize hyperparameters and address overfitting.",
                    "priority": "High",
                })
                used.add("optimize")

        if deployment_status.get("deployed"):
            recommendations.append({
                "recommendation": "Continue monitoring production performance and drift.",
                "priority": "Medium",
            })
            used.add("monitor")
        elif model_history and any(entry.get("test_score") is not None for entry in model_history):
            recommendations.append({
                "recommendation": "Deploy the current best model to production.",
                "priority": "Medium",
            })
            used.add("deploy")

        if not recommendations:
            recommendations.append({
                "recommendation": "Continue model development and validate results.",
                "priority": "Low",
            })

        priority_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        return sorted(recommendations, key=lambda item: priority_order.get(item["priority"], 4))

    def delegate_project_tasks(
        self,
        project_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        project_summary = project_summary or {}
        tasks: List[str] = []
        project_stage = project_summary.get("project_stage", "Unknown")
        risks = project_summary.get("risks", [])
        next_action = project_summary.get("next_best_action", "")

        if "cleaning" in next_action.lower() or any("missing" in risk.lower() for risk in risks):
            tasks.extend(["DatasetIntelligenceAgent", "FeatureEngineeringAgent"])
        if "overfitting" in next_action.lower() or "optimize hyperparameters" in next_action.lower():
            tasks.extend(["SelfHealingAgent", "HyperparameterOptimizationAgent"])
        if "retrain" in next_action.lower() or any("drift" in risk.lower() for risk in risks):
            tasks.append("AutonomousRetrainingAgent")
        if "deploy" in next_action.lower() and "DeploymentAgent" not in tasks:
            tasks.append("DeploymentAgent")
        if not tasks and project_stage in {"Model Optimization", "Production Monitoring"}:
            tasks.append("AIWorkflowAgent")

        if not tasks:
            tasks = ["DatasetIntelligenceAgent", "AIWorkflowAgent"]

        plan = {
            "project_stage": project_stage,
            "ordered_agents": tasks,
            "reasoning": f"Delegated tasks based on project stage '{project_stage}' and identified risks.",
        }

        self._append_decision(
            project_stage=project_stage,
            decision="Delegate project tasks",
            reasoning=plan["reasoning"],
            agents_involved=tasks,
            expected_outcome="Coordinate agents to resolve key issues and advance the ML lifecycle.",
        )

        return plan

    def calculate_project_health(
        self,
        dataset_metadata: Optional[Dict[str, Any]] = None,
        model_history: Optional[List[Dict[str, Any]]] = None,
        drift_history: Optional[List[Dict[str, Any]]] = None,
        deployment_status: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        dataset_metadata = dataset_metadata or {}
        model_history = model_history or []
        drift_history = drift_history or []
        deployment_status = deployment_status or {}

        missing_pct = float(dataset_metadata.get("missing_percent", 0.0))
        duplicate_pct = float(dataset_metadata.get("duplicate_percent", 0.0))
        data_quality = max(0.0, 100.0 - missing_pct * 1.5 - duplicate_pct * 2.0)

        if model_history:
            best_score = max((entry.get("test_score") or 0.0) for entry in model_history)
            model_performance = min(100.0, max(0.0, best_score * 100.0))
        else:
            model_performance = 30.0

        explainability = 100.0 if deployment_status.get("shap_explained") else 60.0
        deployment_ready = 100.0 if deployment_status.get("deployed") else (80.0 if model_history else 40.0)

        if drift_history:
            last_drift = drift_history[-1].get("severity", "Low")
            monitoring_status = 100.0 if last_drift == "Low" else (70.0 if last_drift == "Medium" else 40.0)
        else:
            monitoring_status = 60.0

        experiment_maturity = min(100.0, len(model_history) * 10.0)

        score = (
            data_quality * 0.25
            + model_performance * 0.25
            + explainability * 0.15
            + deployment_ready * 0.15
            + monitoring_status * 0.10
            + experiment_maturity * 0.10
        )
        score = float(round(max(0.0, min(score, 100.0)), 1))

        if score >= 90:
            label = "Enterprise Ready"
        elif score >= 75:
            label = "Production Ready"
        elif score >= 50:
            label = "Needs Improvement"
        else:
            label = "High Risk"

        return {
            "score": score,
            "label": label,
            "components": {
                "data_quality": float(round(data_quality, 1)),
                "model_performance": float(round(model_performance, 1)),
                "explainability": float(round(explainability, 1)),
                "deployment_readiness": float(round(deployment_ready, 1)),
                "monitoring_status": float(round(monitoring_status, 1)),
                "experiment_maturity": float(round(experiment_maturity, 1)),
            },
        }

    def generate_executive_report(
        self,
        project_summary: Dict[str, Any],
        strategic_advice: List[Dict[str, Any]],
        delegation_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "title": "Chief Data Scientist Executive Report",
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "project_stage": project_summary.get("project_stage", "Unknown"),
            "project_health": project_summary.get("project_health", 0),
            "health_label": project_summary.get("health_label", "Unknown"),
            "summary": {
                "strengths": project_summary.get("strengths", []),
                "weaknesses": project_summary.get("weaknesses", []),
                "risks": project_summary.get("risks", []),
                "next_best_action": project_summary.get("next_best_action", "Continue monitoring."),
            },
            "recommendations": strategic_advice,
            "execution_plan": delegation_plan,
            "confidence": "High" if project_summary.get("project_health", 0) >= 75 else "Moderate",
            "readiness": project_summary.get("health_label", "Unknown"),
            "notes": [
                "Focus on business impact and operational readiness.",
                "Use the ordered agent plan to execute the next phase.",
            ],
        }

    def simulate_autonomous_cycle(
        self,
        dataset: pd.DataFrame,
        dataset_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        dataset_metadata = dataset_metadata or {}
        steps: List[Dict[str, Any]] = []
        timestamp = datetime.datetime.utcnow()

        steps.append(
            {
                "timestamp": timestamp.isoformat() + "Z",
                "step": "Dataset arrival",
                "description": "A new dataset has been detected and ingested into the ML lifecycle.",
            }
        )

        self.dataset_agent = DatasetIntelligenceAgent(dataset, dataset_metadata.get("name", "Untitled Dataset"))
        analysis = self.dataset_agent.analyze_problem()
        risks = self.dataset_agent.assess_risks()
        self.feature_agent = FeatureEngineeringAgent(dataset, dataset_metadata.get("name", "Untitled Dataset"))
        feature_plan = self.feature_agent.generate_feature_plan()

        steps.append(
            {
                "timestamp": (timestamp + datetime.timedelta(seconds=5)).isoformat() + "Z",
                "step": "Analyze dataset",
                "description": f"Dataset intelligence identified the problem as {analysis.get('problem_type')}.",
            }
        )

        if risks.get("missing_values", 0.0) >= 2.0 or risks.get("duplicates", 0.0) >= 1.0:
            steps.append(
                {
                    "timestamp": (timestamp + datetime.timedelta(seconds=10)).isoformat() + "Z",
                    "step": "Assess data quality",
                    "description": "Detected data quality issues requiring cleaning and imputation.",
                }
            )

        steps.append(
            {
                "timestamp": (timestamp + datetime.timedelta(seconds=15)).isoformat() + "Z",
                "step": "Generate feature engineering plan",
                "description": feature_plan.get("steps", []),
            }
        )

        steps.append(
            {
                "timestamp": (timestamp + datetime.timedelta(seconds=20)).isoformat() + "Z",
                "step": "Train models",
                "description": f"Baseline and candidate models are selected for {analysis.get('problem_type')}.",
            }
        )

        steps.append(
            {
                "timestamp": (timestamp + datetime.timedelta(seconds=25)).isoformat() + "Z",
                "step": "Optimize hyperparameters",
                "description": "Hyperparameter optimization is planned to improve model performance.",
            }
        )

        steps.append(
            {
                "timestamp": (timestamp + datetime.timedelta(seconds=30)).isoformat() + "Z",
                "step": "Explain model",
                "description": "Feature explainability is generated to validate model decisions.",
            }
        )

        steps.append(
            {
                "timestamp": (timestamp + datetime.timedelta(seconds=35)).isoformat() + "Z",
                "step": "Prepare deployment",
                "description": "The model package is prepared and reviewed for production readiness.",
            }
        )

        steps.append(
            {
                "timestamp": (timestamp + datetime.timedelta(seconds=40)).isoformat() + "Z",
                "step": "Monitor drift",
                "description": "Data drift monitoring is activated to track distribution changes in production.",
            }
        )

        steps.append(
            {
                "timestamp": (timestamp + datetime.timedelta(seconds=45)).isoformat() + "Z",
                "step": "Retrain when needed",
                "description": "The system will retrain automatically if drift or performance decay is detected.",
            }
        )

        return steps

    def get_decision_history(self) -> List[Dict[str, Any]]:
        return list(self.decisions)

    def _infer_project_stage(
        self,
        dataset: pd.DataFrame,
        dataset_metadata: Dict[str, Any],
        model_history: List[Dict[str, Any]],
        drift_history: List[Dict[str, Any]],
        deployment_status: Dict[str, Any],
        workflow_status: Dict[str, Any],
    ) -> str:
        if deployment_status.get("deployed"):
            if drift_history and any(entry.get("severity") == "High" for entry in drift_history):
                return "Production Monitoring"
            return "Production"

        if model_history:
            if any(entry.get("train_score") is not None and entry.get("test_score") is not None and entry.get("train_score") - entry.get("test_score", 0.0) >= 0.15 for entry in model_history):
                return "Model Optimization"
            return "Model Validation"

        missing_pct = float(dataset_metadata.get("missing_percent", 0.0))
        if missing_pct > 10.0:
            return "Data Preparation"

        if workflow_status.get("current_stage") in {"explainability_completed", "model_trained", "report_generated"}:
            return "Model Development"

        return "Exploratory Analysis"

    def _summarize_strengths(
        self,
        dataset_report: Dict[str, Any],
        model_history: List[Dict[str, Any]],
        deployment_status: Dict[str, Any],
    ) -> List[str]:
        strengths: List[str] = []
        domain = dataset_report.get("domain_analysis", {}).get("domain")
        if domain:
            strengths.append(f"Dataset aligned with domain: {domain}.")
        if model_history:
            best = max(model_history, key=lambda item: item.get("test_score") or 0.0)
            if best.get("test_score") is not None:
                strengths.append(f"Top experiment has test score {best['test_score']:.2f}.")
        if deployment_status.get("deployed"):
            strengths.append("Deployment pipeline is available.")
        return strengths

    def _summarize_weaknesses(
        self,
        risk_assessment: Dict[str, Any],
        deployment_status: Dict[str, Any],
    ) -> List[str]:
        weaknesses = []
        if risk_assessment.get("missing_values", 0.0) >= 2.0:
            weaknesses.append("Data quality issues require cleaning and imputation.")
        if risk_assessment.get("duplicates", 0.0) >= 1.0:
            weaknesses.append("Duplicate records may bias model training.")
        if not deployment_status.get("deployed"):
            weaknesses.append("No production deployment is available yet.")
        return weaknesses

    def _summarize_risks(self, risk_assessment: Dict[str, Any], drift_history: List[Dict[str, Any]]) -> List[str]:
        risks = []
        risks.extend(risk_assessment.get("risks", []) if isinstance(risk_assessment.get("risks"), list) else [])
        if drift_history and any(entry.get("severity") == "High" for entry in drift_history):
            risks.append("High data drift requires immediate retraining.")
        if drift_history and any(entry.get("severity") == "Medium" for entry in drift_history):
            risks.append("Moderate drift may impact model stability.")
        return risks
